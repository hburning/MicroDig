import time
import os
import datetime
import json
import pandas as pd
import numpy as np
import sys
from typing import List, Dict, Set, Tuple



TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 默认的时间格式


def ts2date(ts):  # , format=TIME_FORMAT):
    return datetime.datetime.fromtimestamp(ts)
    #time.strftime(format, time.localtime(ts))

def date2datetime(s):
    return datetime.datetime.strptime(s, TIME_FORMAT)

def date2ts(date, format=TIME_FORMAT):
    return int(time.mktime(time.strptime(date, format)))


def ts2date_for_df(df, time_col, date_col):
    df[date_col] = df[time_col].map(datetime.datetime.fromtimestamp)
# pd.Timestamp('2020-01-02 12:30', tz='Asia/Shanghai')


def load_json(json_file):
    if os.path.exists(json_file):
        return json.load(open(json_file, 'r'))
    else:
        print(f'No such file: {json_file}')


def cut_df(df: pd.DataFrame, start: int, end: int, at: str):
    return df[(df[at] >= start) & (df[at] <= end)]
