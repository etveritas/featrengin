# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create on: 2019-04-09 14:42
# LUX et VERITAS

# Import modules
import datetime

import CONSTANT
from util import log, timeit

# Function: 
@timeit
def clean_tables(tables):
    for tname in tables:
        log(f"cleaning table {tname}")
        clean_df(tables[tname])

# Function:
@timeit
def clean_df(df):
    fillna(df)

# Function:
@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)
    
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)
    
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)
    
    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c].fillna("0", inplace=True)

# Function:
@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df, config)

# Function:
@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c+"_"+"year"] = df[c].map(lambda tdata: tdata.year)
        df[c+"_"+"month"] = df[c].map(lambda tdata: tdata.month)
        df[c+"_"+"day"] = df[c].map(lambda tdata: tdata.day)
        df[c+"_"+"hour"] = df[c].map(lambda tdata: tdata.hour)
        df[c+"_"+"minute"] = df[c].map(lambda tdata: tdata.minute)
        df.drop(c, axis=1, inplace=True)

# Function:
@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))
    
    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c+"_"+"cate"] = df[c].apply(lambda x: len(x.split(',')))
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
