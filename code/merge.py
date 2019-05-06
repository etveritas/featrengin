import os
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

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
        agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                     and not col.startswith(CONSTANT.TIME_PREFIX)
                     and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}
        v = v.groupby(key).agg(agg_funcs)
        v.columns = v.columns.map(lambda a:
                f"{CONSTANT.NUMERICAL_PREFIX}{a[1].upper()}({a[0]})")
    else:
        v = v.set_index(key)
    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)



# Function: temporal series combination, the major function
@timeit
def temporal_join(u, v, v_name, key, time_col):
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

    agg_funcs = {col: Config.aggregate_op(col) for col in v if col != key
                 and not col.startswith(CONSTANT.TIME_PREFIX)
                 and not col.startswith(CONSTANT.MULTI_CAT_PREFIX)}

    # TODO(etveritas): accelerate the speed of group calculation
    tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(agg_funcs)
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

    return ret


# Function: Using Depth-First Search to merge tables
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
            u = temporal_join(u, v, v_name, key, config['time_col'])
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
