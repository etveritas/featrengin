import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")

import copy
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp

from automl import predict, train, validate
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table, proc_sltable 
from preprocess import clean_df, clean_tables, feature_engineer
from util import Config, log, show_dataframe, timeit

# Class: entrance class
class Model:
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)
        self.y = copy.deepcopy(y)

    @timeit
    def predict(self, X_test, time_remain):
        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f"{x[0]}_{x[1]}")
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)
        X = merge_table(Xs, self.config)
        del Xs
        clean_df(X)
        feature_engineer(X, self.config)
        proc_sltable(X)
        X_train = X[X.index.str.startswith("train")]
        X_test = X[X.index.str.startswith("test")]
        # X_train.to_csv("XA_train.csv", index=False)
        # self.y.to_csv("YA_train.csv", index=False)
        del X
        gc.collect()
        X_train.index = X_train.index.map(lambda x: int(x.split("_")[1]))
        X_test.index = X_test.index.map(lambda x: int(x.split('_')[1]))
        X_train.sort_index(inplace=True)
        X_test.sort_index(inplace=True)
        train(X_train, self.y, self.config)
        result = predict(X_test, self.config)

        return pd.Series(result)
