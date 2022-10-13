import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import os
import glob
from tqdm import tqdm, trange
import pickle
import glob

from anomaly import AnomalyDetector
from graph import GraphGen
from ranker import Ranker
from evaluator import Evaluator
from dataloader import CaseModel

import signal
def handler(signum, addition):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
signal.signal(signal.SIGTERM, handler)


def normalization(data):
    if len(data) == 0:
        return data
    if isinstance(data, list):
        data = np.array(data)
    _range = np.max(data) - np.min(data)
    if _range == 0:
        print(data)
    res = (data - np.min(data)) / _range
    return list(res) if isinstance(data, list) else res


hyper_params = {
    'test_length_before': 10, 
    'test_length_after': 10,
    'train_length': 60,
    'search_method': 'all',  
    'rank_method': 'random walk',  # [pagerank, random walk]
    'level': 'method', 
    'rev_weight': 0.2,   # weight of the reverse edges
    'server_data': False,
    'beta': 0.1,
    'new_col': 'final_score',
}


anomaly_detector = AnomalyDetector(hyper_params)
out_dir = './figures'
plot = True
level = hyper_params['level']

if __name__ == '__main__':

    df = pd.read_csv('../../cases_testbed.csv')
    info_dict = {}
    for index, row in df.iterrows():
        case_name = row['timestamp']
        root_cause = [row['root_cause_node']]
        alarm_item = row['alarm_item']
        exp_type = row['experiment_type']
        info_dict[case_name] = {
            'exp_type': exp_type,
            'root_cause': root_cause,
            'alarm_item': alarm_item
        }

    case_paths = glob.glob('../../cases/*.pkl')
    case_paths = sorted(case_paths)
    os.makedirs(out_dir, exist_ok=True)

    evaluator1 = Evaluator('./results-test-alg1')
    evaluator4 = Evaluator('./results-test-alg4')
    evaluator5 = Evaluator('./results-test-alg5')

    for path in tqdm(case_paths):
        print(f'\n\nLoading case: {path}')
        case = pickle.load(open(path, 'rb'))
        case_name = path.split('/')[-1][:-4]
        case.rc = info_dict[case_name]['root_cause']
        case.alarm_item = info_dict[case_name]['alarm_item']
        
        graph_gen = GraphGen(anomaly_detector, hyper_params, case)
        ranker = Ranker(hyper_params, case, graph_gen)

        ## Ranking
        res_alg1_server, res_alg1_method = ranker.alg1()
        if res_alg1_method.empty:
            continue

        graph_gen.method_score_df = res_alg1_method
        res_alg4 = ranker.mix_node_rank_wodetect()
        res_alg5 = ranker.server_node_rank()

        ## Plot
        if plot:
            figure_out = os.path.join(out_dir, case.alarm_start_time)
            os.makedirs(figure_out, exist_ok=True)

            G_ano = graph_gen.get_asso_method_node_ano_graph()
            graph_gen.draw_related_callings(case.callings, G_ano, figure_out)

            graph_gen.draw_mix_graph(figure_out)
            graph_gen.draw_server_node_graph(figure_out)
            
        ## judgement
        if res_alg4 is not None and res_alg4.shape[0] == 0:
            print(f'Can not find candidate root cause: {case.alarm_start_time}')
            continue
        
        ## evaluation
        mar1_server = evaluator1.process_result_server(res_alg1_server, case, score_col = 'server_score_norm')
        mar1_method = evaluator1.process_result_method(res_alg1_method, case, score_col = 'final_score')
        mar4 = evaluator4.process_result_server(res_alg4, case, score_col = 'final_score')
        mar5 = evaluator5.process_result_server(res_alg5, case, score_col = 'final_score')
        
        print(f'\nCase:{case.alarm_start_time},\n alg1 mar:{mar1_server}, alg4 mar:{mar4}, alg5 mar:{mar5}')

    evaluator1.save_final_result()
    evaluator1.save_final_result(tp = hyper_params['level'])
    evaluator4.save_final_result()
    evaluator5.save_final_result()
