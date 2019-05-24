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
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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
def join(u, v, u_name, v_name, key, type_):
    ckey = ",".join(key)
    # preprocess the TIME and MULTI_CAT columns
    time_cols = [col for col in v if col.startswith(CONSTANT.TIME_PREFIX)]
    multi_cat_cols = [col for col in v if col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.minute).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_minute", time_cols))))], axis=1)
    v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.second).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_second", time_cols))))], axis=1)
    v = pd.concat([v, v[time_cols].apply(lambda x: x.dt.microsecond).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_microsecond", time_cols))))], axis=1)
    v = pd.concat([v, v[multi_cat_cols].apply(lambda x: x.str.count(",")).rename(columns=dict(zip(multi_cat_cols, map(lambda x: f"mn_{x}", multi_cat_cols))))], axis=1)
    if type_.split("_")[2] == 'many':
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)}

        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a:
                f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]})")
    else:
        # proc_sltable(v, key)
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{u_name}_{ckey}_{v_name}_{a}")

    return u.join(v, on=key)
    # return v


def nprolling(tdata, window):
    comp = np.full(tdata.shape[:-1]+(window-1, ), np.nan)
    tdata = np.concatenate((comp, tdata), axis=-1)
    shape = tdata.shape[:-1] + (tdata.shape[-1] - window + 1, window)
    strides = tdata.strides + (tdata.strides[-1],)

    return np.lib.stride_tricks.as_strided(tdata, shape=shape, strides=strides)

def eOpsS(OpData, Op, roll_win, ood):
    timer = Timer()
    res_val = np.array([])
    res_col = np.empty((1,), dtype=object)
    op_key = list(Op.keys())[0]
    op_op = list(Op.values())[0][0]
    res_col[:] = [(op_key, op_op)]
    for arr in OpData:
        rollarr = nprolling(arr, roll_win)
        if op_op == "count":
            res_val = np.concatenate((res_val, np.count_nonzero(rollarr == rollarr, axis=1)))
        elif op_op == "max":
            res_val = np.concatenate((res_val, np.max(rollarr, axis=1)))
        elif op_op == "min":
            res_val = np.concatenate((res_val, np.min(rollarr, axis=1)))
        elif op_op == "mean":
            res_val = np.concatenate((res_val, np.mean(rollarr, axis=1)))
        elif op_op == "sum":
            res_val = np.concatenate((res_val, np.sum(rollarr, axis=1)))

    # res_val = res_val.reshape(-1, 1)

    # recover orginal order
    res_val = res_val[ood]

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
        ckey = key[0]

    rows_u = u.shape[0]
    # rows_v = v.shape[0]

    tmp_u = u[[time_col, ckey]]
    timer.check("select")

    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    timer.check("concat")

    rehash_key = f'rehash_{ckey}'
    tmp_u[rehash_key] = tmp_u[ckey].apply(lambda x: hash(x) % CONSTANT.HASH_MAX)
    timer.check("rehash_key")

    tmp_u["ood"] = np.arange(tmp_u.shape[0])
    tmp_u.sort_values(time_col, inplace=True)
    ood = np.argsort(tmp_u["ood"].values)
    tmp_u.drop("ood", axis=1, inplace=True)
    timer.check("sort")

    # preprocess the TIME and MULTI_CAT columns
    time_cols = [col for col in tmp_u if col.startswith(CONSTANT.TIME_PREFIX)]
    multi_cat_cols = [col for col in tmp_u if col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.minute).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_minute", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.second).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_second", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[time_cols].apply(lambda x: x.dt.microsecond).rename(columns=dict(zip(time_cols, map(lambda x: f"tn_{x}_microsecond", time_cols))))], axis=1)
    tmp_u = pd.concat([tmp_u, tmp_u[multi_cat_cols].apply(lambda x: x.str.count(",")).rename(columns=dict(zip(multi_cat_cols, map(lambda x: f"mn_{x}", multi_cat_cols))))], axis=1)

    agg_funcs = {col: Config.aggregate_op(col) for col in tmp_u if col != ckey
                 and col != rehash_key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                }

    listOps = list()
    for key, values in agg_funcs.items():
        for op in values:
            listOps.append({key: [op]})
    log(f"[Operation numbers] {len(listOps)}")

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
    
    del tmp_u
    timer.check("pre-rolling")

    # rolling windows
    roll_win = 30

    # tmp_u = pd.DataFrame(index=pd.MultiIndex.from_tuples(tmp_u_idx))
    # for Op in listOps:
    #     col = list(Op.keys())[0]
    #     resCol, resVal = eOpsS(scol_dict[col], Op, roll_win)
    #     tmp_u[resCol[0]] = resVal

    for Op in listOps:
        col = list(Op.keys())[0]
        resCol, resVal = eOpsS(scol_dict[col], Op, roll_win, ood)
        n_col = f"{CONSTANT.NUMERICAL_PREFIX}{resCol[0][1].upper()}_ROLLING{roll_win}({u_name}_{ckey}_{v_name}_{resCol[0][0]})"
        if resVal.size != 0:
            u[n_col] = resVal[: rows_u]

    # # tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
    timer.check("group & rolling & agg")

    # tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index
    # timer.check("reset_index")

    # tmp_u.columns = tmp_u.columns.map(lambda a:
    #     f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}_ROLLING{roll_win}({u_name}_{ckey}_{v_name}_{a[0]})")

    # if tmp_u.empty:
    #     log("empty tmp_u, return u")
    #     return u

    # ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    # timer.check("final concat")
    # del tmp_u

    # return ret

    # if not tmp_u.empty:
    #     return tmp_u.loc['u']
    # else:
    #     return

    return u


@timeit
def proc_sltable(df, key=[]):
    """
    """
    
    for col in [col for col in df if col.startswith(CONSTANT.CATEGORY_PREFIX) and col not in key]:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    for col in [col for col in df if col.startswith(CONSTANT.MULTI_CAT_PREFIX) and col not in key]:
        n_cp = 10
        tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_vec = tfidf_enc.fit_transform(df[col])
        svd_enc = TruncatedSVD(n_components=n_cp, n_iter=20, random_state=1)
        mode_svd = svd_enc.fit_transform(tfidf_vec)
        for idx in range(n_cp):
            df[f"n_{col}_tfidf_{idx}"] = mode_svd[:, idx]
        df.drop(col, axis=1, inplace=True)
        


# Function: Using Depth-First Search to merge tables
# TODO(etveritas): use iteration not recursion if need
def dfs(u_name, config, tables, graph):
    """
    In graph, every pair of nodes(two tables) has "From" and "To", the process is 
    combining "To" to "From", so "From" is the major table and what we get.

    The rules of combination are follows:
    1. Combine when depth of "From" is greater than "To"'s (base rule).
    2. Combine when not "From" has no major time, however "To" has major time.
    """

    # depth_map = defaultdict(list)
    # for e_name in config["tables"].keys():
    #     curr_depth = config["tables"][e_name]["depth"]
    #     depth_map[curr_depth].append(e_name)
    
    # for depthi in reversed(range(max(depth_map.keys()))):
    #     for f_name in depth_map[depthi]:
    #         f_data = tables[f_name]
    #         feated_dict = {"t": list(), "nt": list()}
    #         for t_edge in graph[f_name]:
    #             t_name = t_edge['to']
    #             log(f"t_name: {t_name}")
    #             if config['tables'][t_name]['depth'] <= config['tables'][f_name]['depth']:
    #                 continue
    #             t_data = tables[t_name]
    #             key = t_edge['key']
    #             type_ = t_edge['type']
    #             if config["time_col"] not in f_data and config["time_col"] in t_data:
    #                 continue
    #             if config["time_col"] in f_data and config["time_col"] in t_data:
    #                 log(f"join {f_name} <--{type_}--t {t_name}")
    #                 feated_dict["t"].append(temporal_join(f_data, t_data, f_name, t_name, key, config['time_col'], type_))
    #             else:
    #                 log(f"join {f_name} <--{type_}--nt {t_name}")
    #                 feated_dict["nt"].append((key, join(f_data, t_data, f_name, t_name, key, type_)))

    #         for con_type in feated_dict.keys():
    #             for edata in feated_dict[con_type]:
    #                 if con_type == "t":
    #                     tables[f_name] = pd.concat([tables[f_name], edata], axis=1, sort=False)
    #                 elif con_type == "nt":
    #                     tables[f_name] = tables[f_name].join(edata[1], on=edata[0])
    #                 del edata

    # return tables[u_name]

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
            u = join(u, v, u_name, v_name, key, type_)

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
