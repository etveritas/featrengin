# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create on: 2019-04-09 10:32
# LUX et VERITAS

"""
This module is util,
including functions and classes for auto process.
"""

# Import modules
import os 
import pickle 
import time 
from typing import Any 

import CONSTANT 


# global variables to indicate some thing
nesting_level = 0
is_start = None

# Class: Timer, to check time info
class Timer:
    def __init__(self):
        """
        Initialization function.
        """
        self.start = time.time()
        self.history = [self.start]
    
    def check(self, info):
        """
        Get current running time and push this time point into history list.
        """
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

# Function: to indicate time pass, is an decorator
def timeit(method, start_log=None):
    """
    Paramters:
    method: callable type, need to calculate thie passing time.
    start_log: default None

    Returns:
    timed: callable function
    """

    def timed(*args, **kw):
        """
        Decorator function do really.
        Can recieve any parameters.
        """

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

# Function: log function
def log(entry: Any):
    """
    Print log message on console.
    Parameters:
    entry: Any type, wait to print
    """

    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

# Function: show dataframe
def show_dataframe(df):
    """
    Parameters:
    df: pandas DataFrame, to show
    """
    
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

# Class: Config, define config
class Config:
    def __init__(self, info):
        """
        Parameters:
        info: dictionary, including information of configuration.
        """

        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info["tables"].items():
            self.data["tables"][tname] = {}
            self.data["tables"][tname]["type"] = ttype
    
    @staticmethod
    def aggregate_op(col):
        """
        Operation set
        """

        import numpy as np 

        def my_nunique(x):
            return x.nunique()
        
        my_nunique.__name__ = "nunique"
        ops = {
            CONSTANT.NUMERICAL_TYPE: ["mean", "sum", "max", "min", "std"],
            CONSTANT.CATEGORY_TYPE: ["count"],
        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.MULTI_CAT_PREFIX):
            assert False, f"MultiCategory type feature's aggregate op are not supported."
            return ops[CONSTANT.MULTI_CAT_TYPE]
        if col.startswith(CONSTANT.TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."
        assert False, f"Unknown col type {col}"
    
    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])
    
    # some special functions
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
