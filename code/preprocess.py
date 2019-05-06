import datetime

import CONSTANT
from util import log, timeit
from collections import Counter 


# Function: clean tables
@timeit
def clean_tables(tables):
    """
    Clean tables by handling missing value and abnormal value for now.

    Parameters:
    tables: dictionary, which key is the table name and value is the table value storing in DataFrame.
    """

    for tname in tables:
        log(f"cleaning table {tname}")
        hdabval(tables[tname])
        hdmsval(tables[tname])


# Function: clean DataFrame
@timeit
def clean_df(df):
    """
    """
    hdmsval(df)


# Function: handle missing value
@timeit
def hdmsval(df):
    """
    Handle the missing value depend on the data type of value.
    For now, we just do the ordinary fill, somthing like, 0 for
    numerical value, "0" for value of string, UNIX orginal time 
    for time.

    Parameters:
    df: pandas DataFrame, dataset waits for filling missing value.
    """

    NumSet = [col for col in df if col.startswith(CONSTANT.NUMERICAL_PREFIX)]
    StrSet = [col for col in df if col.startswith(CONSTANT.CATEGORY_PREFIX) or \
              col.startswith(CONSTANT.MULTI_CAT_PREFIX)]
    TimeSet = [col for col in df if col.startswith(CONSTANT.TIME_PREFIX)]

    if NumSet:
        df[NumSet] = df[NumSet].fillna(-1)
    if StrSet:
        df[StrSet] = df[StrSet].fillna("0")
    if TimeSet:
        df[TimeSet] = df[TimeSet].fillna(datetime.datetime(1970, 1, 1))


# Function: handle abnormal value
@timeit
def hdabval(df):
    """
    We handle abnormal value by calculating the quantile values.
    We fill abnormal value with median depend on 0.75 quantile and 0.25 quantile value.

    Equation: a.b = (N-1)*QFrac, Pos = a + 1, QVal = X[Pos-1] + (X[Pos] - X[Pos-1])*b.
    Rule: If x > QVal0.75 + 1.5d or x < QVal0.25 - 1.5d Then x = Qval0.5.
    N: total numbers, excluding NULL value; 
    QFrac: quantile fraction; 
    a: the integral part (if 2.3 then a euqals 2); 
    b: the fractional part (if 2.3 then b euqals 0.3); 
    Pos: lower position which the quantile value lies; 
    QVal: quantile value;
    x: feature value;
    d: QVal0.75-QVal0.25.

    Note: we only handle the numerical value, and calculate quantile by linear method.

    Parameters:
    df: pandas DataFrame, dataset waits for handling abnormal value.
    """

    # get numerical columnes set
    NumSet = [col for col in df if col.startswith(CONSTANT.NUMERICAL_PREFIX)]
    if NumSet: 
        # calculate the quantile values and save it to a DataFrame
        Qdf = df[NumSet].quantile([.25, .75, .5])
        # for the convenience of operation we use "1.5" to represent the "delta"(i.e. QVal0.75-QVal0.5)
        Qdf.loc[1.5] = Qdf.loc[0.75] - Qdf.loc[0.25]
        # translate
        # Note: it's a bad idea to handle anything by "apply" function, 
        # we should use matrix(vector) operations to accelerate as much as possible in Python.
        df[NumSet] = df[NumSet][(df[NumSet] > Qdf.loc[.75] + 1.5*Qdf.loc[1.5]) | \
                     (df[NumSet] < Qdf.loc[.25] - 1.5*Qdf.loc[1.5])].fillna(Qdf.loc[.5])


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df, config)


@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c+"_month"] = df[c].apply(lambda x: x.month)
        df[c+"_day"] = df[c].apply(lambda x: x.day)
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(CONSTANT.MULTI_CAT_PREFIX)]:
        df[c+"_max"] = df[c].apply(lambda x: len(x.split(",")))
        df[c+"_num"] = df[c].apply(lambda x: int(x.split(",")[0]))
        df.drop(c, axis=1, inplace=True)