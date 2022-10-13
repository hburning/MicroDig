import numpy as np
import pandas as pd
from typing import List

class AnomalyDetector:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.anomaly_score_func = AnomalyDetector.ksigma

    @staticmethod
    def ksigma(train: np.ndarray, test: np.ndarray, is_alarm: bool) -> float:
        eps = 1e-3
        mean = train.mean()
        std = train.std()
        mmax = test.max()
        mmin = test.min()
        if (mmax - mmin < 0.0005) or (mmax <= train.max()):
        # if (mmax - mmin < 0.005):
            if is_alarm:
                return 0.01
            else:
                return 0
        temp_score = []
        for x in test:
            if ((x > mean+3*std) or (x < mean-3*std)) and (abs(x-mean) >= 0.0005):
            # if ((x > mean+3*std) or (x < mean-3*std)):
                # temp_score.append(np.log(1 + abs(x-mean)/(std+eps)))
                temp_score.append(1)
                break
            else:
                if is_alarm:
                    temp_score.append(0.01)
                else:
                    temp_score.append(0)
        return max(temp_score)

    def edge_score(self, alarm_minute: int, data_list: List, is_alarm: bool) -> List:
        train_length = self.hyper_params['train_length']
        test_length_before = self.hyper_params['test_length_before']
        test_length_after = self.hyper_params['test_length_after']
        scores = []
        for d in data_list:
            df = pd.DataFrame({'min': d.keys(), 'value': d.values()})
            train = df[(df['min'] > alarm_minute - test_length_before - train_length)
                       & (df['min'] <= alarm_minute - test_length_before)]['value'].values
            test = df[(df['min'] > alarm_minute - test_length_before) &
                      (df['min'] <= alarm_minute + test_length_after)]['value'].values
            if train.shape[0] < 5 or test.shape[0] == 0:
                return [0]*len(data_list)

            scores.append(self.anomaly_score_func(train, test, is_alarm))
        return scores
