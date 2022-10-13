import enum
import os
import datetime
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
from scipy.stats import rankdata
from utils import ts2date, date2ts, date2datetime
import pickle
import re 
from collections import defaultdict

time_begin = datetime.datetime(2022,5,27)

class CaseModel:

    def __init__(self, alarm_start_time: datetime.datetime, alarm_end_time: str, monitor_id: int, alarm_item: str, sli_type:str, where_info: Dict):
        self.callings: Dict = {}
        self.alarm_callings: Dict = {}
        self.alarm_start_time: str = str(alarm_start_time)
        self.alarm_end_time: str = str(alarm_end_time)
        self.alarm_start_minute: int = alarm_start_time.hour * 60 + alarm_start_time.minute
        self.alarm_end_minute: int = alarm_end_time.hour * 60 + alarm_end_time.minute if alarm_end_time!='' else 0
        self.monitor_id: int = monitor_id
        self.alarm_item: str = alarm_item
        self.sli_type: str = sli_type
        self.where_info: Dict = where_info
        self.rc: List = [] # root cause
        self.changed: bool = False # whether alarm_item changed
        self.server_num: int = 0


def merge_callings(df: pd.DataFrame, gb_cols: List):
    # merge callings (groupby: gb_cols)
    if df.shape[0] == 0:
        return {}
    min_cols = ['duration', 'error_min', 'error_rate', 'request_min']

    from collections import Counter
    new_data = {}

    # print(df.columns)
    if len(gb_cols):
        groups = df.groupby(gb_cols)
    else:
        groups = [(['all'], df)]

    for i, g in groups:
        key = '|'.join(i)
        temp = [None] * len(min_cols)
        for idx, row in g.iterrows():
            if temp[0]:
                for j in range(len(min_cols)):
                    temp[j].update(Counter(row[min_cols[j]]))
            else:
                for j in range(len(min_cols)):
                    temp[j] = Counter(row[min_cols[j]])

        new_data[key] = {k: dict(sorted(dict(v).items(), key=lambda x: x[0]))
                         for k, v in zip(min_cols, temp)}
        new_data[key]['name'] = key
    return new_data
