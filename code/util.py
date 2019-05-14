import os
import pickle
import time
from typing import Any
import numpy as np 


import CONSTANT

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")


class Config:
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        ops = {
            CONSTANT.NUMERICAL_TYPE: ["mean", "sum", "max", "min", "std", vptp, vkurt, vskew],
            CONSTANT.CATEGORY_TYPE: ["count", vnunique, vmax, vmin, vmean, vfval, vlval],
            CONSTANT.MULTI_CAT_TYPE: ["count", vnunique],
            CONSTANT.MULTI_CAT_NUM_TYPE: ["max", "mean", "min", vmax, vmin, vmean],
            CONSTANT.TIME_NUM_TYPE: [vnunique, "max", "min", "mean", vmax, vmin, vmean, vptp],
        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        elif col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]
        elif col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            return ops[CONSTANT.MULTI_CAT_TYPE]
        # elif col.startswith(CONSTANT.TIME_PREFIX):
        #     return ops[CONSTANT.TIME_TYPE]
        elif col.startswith(CONSTANT.MULTI_CAT_NUM_PREFIX):
            return ops[CONSTANT.MULTI_CAT_NUM_TYPE]
        elif col.startswith(CONSTANT.TIME_NUM_PREFIX):
            return ops[CONSTANT.TIME_NUM_TYPE]
        else:
            assert False, f"Unknown col type {col}"

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

# This part define many tiny functions to replace using lambda direct.
# They all receive Seires
def vptp(Sval):
    return np.ptp(Sval)

def vmax(Sval):
    return Sval.value_counts().max()

def vmin(Sval):
    return Sval.value_counts().min()

def vmean(Sval):
    return Sval.value_counts().mean()

def vkurt(Sval):
    return Sval.kurt()

def vskew(Sval):
    return Sval.skew()

def vnunique(Sval):
    return Sval.nunique()

def vfval(Sval):
    return int(Sval.value_counts().index[0])

def vlval(Sval):
    return int(Sval.value_counts().index[-1])
