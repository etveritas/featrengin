import os
import gc
import time
import pickle 
from collections import defaultdict, deque
import copy
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp 

import CONSTANT
from util import Config, Timer, log, timeit

NUM_OP = [np.std, np.mean]


# Function: Using Breadth-First Search to calculate the depth of relation graph
def bfs(root_name, graph, tconfig):
    """
    We use the method of baseline to get the relations between tables.

    Parameters:
    root_name: string, name of root node.
    graph: List dictionary, relations of tables.
    tconfig: dictionary, config of tables.

    from root and get the depth. Assume the relation graph like this: 
                MAIN_TABLE
                /        \
               /          \
              /            \
             /              \
          TABLE_1 - - - - TABLE_2
           /
          /
         /
      TABLE_3
    every edge is bidirectional, then depth of MAIN_TABLE is 0, 
    depth of TABLE_1 and TABLE_2 are 1, depth of TABLE_3 is 2.
    """

    tconfig[root_name]['depth'] = 0
    queue = deque([root_name])
    while queue:
        u_name = queue.popleft()
        for edge in graph[u_name]:
            v_name = edge['to']
            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@timeit
def join(u, v, v_name, key, type_):
    if type_.split("_")[2] == 'many':
        # preprocess the TIME and MULTI_CAT columns
        time_cols = [col for col in v if col.startswith(CONSTANT.TIME_PREFIX)]
        multi_cat_cols = [col for col in v if col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
        v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.day).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_day", time_cols))))], axis=1)
        v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.hour).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_hour", time_cols))))], axis=1)
        v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.minute).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_minute", time_cols))))], axis=1)
        v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.second).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_second", time_cols))))], axis=1)
        v = pd.concat([v, v[multi_cat_cols].apply(lambda x: x.str.count(",")).rename(columns=dict(zip(multi_cat_cols, map(lambda x: f"mn_{x}", multi_cat_cols))))], axis=1)

        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)}

        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a:
                f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]})")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


# Function: Target function
# def eOps(OpData, Gkey, Op, Onum, qRes):
#     timer = Timer()
#     fname = f"tmp_{Onum}.pkl"
#     gra = OpData.groupby(Gkey).rolling(5).agg(Op)
#     Res = [gra.columns.values, gra.values]
#     qRes.put(Res)
#     del gra
#     del OpData
#     gc.collect()
#     timer.check(f"Put {list(Op.keys())[0]}-{list(Op.values())[0][0]} in Queue")


def nprolling(tdata, window):
    comp = np.full(tdata.shape[:-1]+(window-1, ), np.nan)
    tdata = np.concatenate((comp, tdata), axis=-1)
    shape = tdata.shape[:-1] + (tdata.shape[-1] - window + 1, window)
    strides = tdata.strides + (tdata.strides[-1],)

    return np.lib.stride_tricks.as_strided(tdata, shape=shape, strides=strides)

def eOps(OpData, Op, qRes):
    timer = Timer()
    res_val = np.array([])
    res_col = np.empty((1,), dtype=object)
    op_key = list(Op.keys())[0]
    op_op = list(Op.values())[0][0]
    res_col[:] = [(op_key, op_op)]
    for arr in OpData:
        rollarr = nprolling(arr, 5)
        if op_op == "count":
            res_val = np.concatenate((res_val, np.sum(rollarr == rollarr, axis=1)))
        elif op_op == "max":
            res_val = np.concatenate((res_val, np.max(rollarr, axis=1)))
        elif op_op == "min":
            res_val = np.concatenate((res_val, np.min(rollarr, axis=1)))
        elif op_op == "mean":
            res_val = np.concatenate((res_val, np.mean(rollarr, axis=1)))
        elif op_op == "sum":
            res_val = np.concatenate((res_val, np.sum(rollarr, axis=1)))
    res_val = res_val.reshape(-1, 1)
    Res = [res_col, res_val]
    qRes.put(Res)
    del OpData
    gc.collect()
    timer.check(f"Put {op_key}-{op_op} in Queue")


def eOpsS(OpData, Op):
    timer = Timer()
    res_val = np.array([])
    res_col = np.empty((1,), dtype=object)
    op_key = list(Op.keys())[0]
    op_op = list(Op.values())[0][0]
    res_col[:] = [(op_key, op_op)]
    for arr in OpData:
        rollarr = nprolling(arr, 5)
        if op_op == "count":
            res_val = np.concatenate((res_val, np.sum(rollarr == rollarr, axis=1)))
        elif op_op == "max":
            res_val = np.concatenate((res_val, np.max(rollarr, axis=1)))
        elif op_op == "min":
            res_val = np.concatenate((res_val, np.min(rollarr, axis=1)))
        elif op_op == "mean":
            res_val = np.concatenate((res_val, np.mean(rollarr, axis=1)))
        elif op_op == "sum":
            res_val = np.concatenate((res_val, np.sum(rollarr, axis=1)))
    res_val = res_val.reshape(-1, 1)

    timer.check(f"Done {op_key}-{op_op}")
    return res_col, res_val
    


# Function: temporal series combination, the major function
@timeit
def temporal_join(u, v, u_name, v_name, key, time_col, type_):
    """
    Do important things in this part. It processes the feature engineering of
    temporal series.

    Parameters:
    u: pandas DataFrame, data of "From" table.
    v: pandas DataFrame, data of "To" table.
    v_name: string, name of "To" table.
    key: list, combination keys, normally, it only has one key.
    time_col: string, main time column's name.
    """

    timer = Timer()

    # judge the length of key, only do when has one key in list
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]
    
    tmp_u = u[[time_col, key]]
    timer.check("select")

    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    timer.check("concat")

    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    timer.check("rehash_key")

    tmp_u.sort_values(time_col, inplace=True)
    timer.check("sort")

    # preprocess the TIME and MULTI_CAT columns
    time_cols = [col for col in tmp_u if col.startswith(CONSTANT.TIME_PREFIX)]
    multi_cat_cols = [col for col in tmp_u if col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.day).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_day", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.hour).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_hour", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.minute).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_minute", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.second).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_second", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[multi_cat_cols].apply(lambda x: x.str.count(",")).rename(columns=dict(zip(multi_cat_cols, map(lambda x: f"mn_{x}", multi_cat_cols))))], axis=1)

    agg_funcs = {col: Config.aggregate_op(col) for col in tmp_u if col != key
                 and col != rehash_key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                }

    # TODO(etveritas): reduce the memory use
    # p_num = mp.cpu_count()//2
    # p_num = 3
    # pool = mp.Pool(p_num)
    # queueRes = mp.Manager().Queue()
    listOps = list()
    
    for key, values in agg_funcs.items():
        for op in values:
            listOps.append({key: [op]})
    log(f"Operation numbers: {len(listOps)}")

    scol_dict = dict()
    tmp_u_idx = np.array([])
    for key in agg_funcs.keys():
        scol_dict[key] = list()
    
    for egroup in tmp_u.groupby(rehash_key):
        eg_idx = np.empty((1,), dtype=object)
        eg_idx[:] = [(egroup[0],)]
        tmp_u_idx = np.concatenate((tmp_u_idx, eg_idx+egroup[1].index.values))
        for key in agg_funcs.keys():
            scol_dict[key].append(egroup[1][key].values)

    for key in agg_funcs.keys():
        scol_dict[key] = np.array(scol_dict[key])
    timer.check("pre-rolling")

    # for Op in listOps:
    #     col = list(Op.keys())[0]
    #     pool.apply_async(eOps, args=(scol_dict[col], Op, queueRes))

    # pool.close()
    # pool.join()

    # tmp_u = pd.DataFrame()
    # log(f"Get Operation numbers: {queueRes.qsize()}")
    # timer.check("group & rolling & agg")


    # ResColTup = ()
    # ResValTup = ()
    # while queueRes.empty() == False:
    #     resCol, resVal = queueRes.get()
    #     ResColTup += (resCol, )
    #     ResValTup += (resVal, )

    tmp_u = pd.DataFrame(index=pd.MultiIndex.from_tuples(tmp_u_idx))
    for Op in listOps:
        col = list(Op.keys())[0]
        resCol, resVal = eOpsS(scol_dict[col], Op)
        tmp_u[resCol[0]] = resVal
    
    # tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    timer.check("group & rolling & agg")

    tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    timer.check("reset_index")

    tmp_u.columns = tmp_u.columns.map(lambda a:
        f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})")

    if tmp_u.empty:
        log("empty tmp_u, return u")
        return u

    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    timer.check("final concat")
    del tmp_u


    # if type_.split("_")[0] == 'many':
    #     orig_u = copy.deepcopy(u)
    #     rehash_key = f'rehash_{key}'
    #     orig_u[rehash_key] = orig_u[key].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    #     timer.check("rehash_key u")

    #     orig_u.sort_values(time_col, inplace=True)
    #     timer.check("sort u")

    #     # preprocess the TIME and MULTI_CAT columns
    #     time_cols = [col for col in orig_u if col.startswith(CONSTANT.TIME_PREFIX)]
    #     multi_cat_cols = [col for col in orig_u if col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    #     orig_u = pd.concat([orig_u, orig_u[time_cols].apply(lambda x: x.dt.day).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_day", time_cols))))], axis=1)
    #     orig_u = pd.concat([orig_u, orig_u[time_cols].apply(lambda x: x.dt.hour).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_hour", time_cols))))], axis=1)
    #     orig_u = pd.concat([orig_u, orig_u[time_cols].apply(lambda x: x.dt.minute).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_minute", time_cols))))], axis=1)
    #     orig_u = pd.concat([orig_u, orig_u[time_cols].apply(lambda x: x.dt.second).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_second", time_cols))))], axis=1)
    #     orig_u = pd.concat([orig_u, orig_u[multi_cat_cols].apply(lambda x: x.str.count(",")).rename(columns=dict(zip(multi_cat_cols, map(lambda x: f"mn_{x}", multi_cat_cols))))], axis=1)

    #     agg_funcs = {col: Config.aggregate_op(col) for col in orig_u if col != key
    #             and col != rehash_key 
    #             and not col.startswith(CONSTANT.TIME_PREFIX)}

    #     # TODO(etveritas): accelerate the speed of group calculation
    #     orig_u = orig_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    #     timer.check("group & rolling & agg u")

    #     orig_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    #     timer.check("reset_index u")

    #     orig_u.columns = orig_u.columns.map(lambda a:
    #         f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({u_name}.{a[0]})")

    #     ret = pd.concat([ret, orig_u], axis=1, sort=False)
    #     timer.check("final concat u")

    #     del orig_u

    return ret


# Function: Using Depth-First Search to merge tables
# TODO(etveritas): use iteration not recursion
def dfs(u_name, config, tables, graph):
    """
    In graph, every pair of nodes(two tables) has "From" and "To", the process is 
    combining "To" to "From", so "From" is the major table and what we get.

    The rules of combination are follows:
    1. Combine when depth of "From" is greater than "To"'s (base rule).
    2. Combine when not "From" has no major time, however "To" has major time.
    """

    u = tables[u_name]
    log(f"enter {u_name}")
    for edge in graph[u_name]:
        v_name = edge['to']
        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            log(f"join {u_name} <--{type_}--t {v_name}")
            u = temporal_join(u, v, u_name, v_name, key, config['time_col'], type_)
        else:
            log(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    log(f"leave {u_name}")
    return u


@timeit
def merge_table(tables, config):
    graph = defaultdict(list)
    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']
        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })
        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })
    bfs(CONSTANT.MAIN_TABLE_NAME, graph, config['tables'])
    return dfs(CONSTANT.MAIN_TABLE_NAME, config, tables, graph)
