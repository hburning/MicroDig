import networkx as nx
from dataloader import CaseModel, merge_callings
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
from graphviz import Digraph


DEPTH = 0
def timeit(comment=None):
    def decorator(func):
        cmt = comment if comment else f'{func.__name__} running...'
        def wrapper(*args, **kwargs):
            global DEPTH
            cmtt = '\n' + '| '*DEPTH + cmt if DEPTH else cmt
            # print(cmtt, end='', flush=True)
            s = time.time()
            e = '\n' if DEPTH == 0 else ' '
            DEPTH += 1
            res = func(*args, **kwargs)
            # print('\n' + '| ' * (DEPTH-1) + f'Done in {time.time() - s:.3f}s', end=e, flush=True)
            DEPTH -= 1
            return res
        return wrapper
    return decorator


def cal_data_rate(d, datakeys):
    error_rate = {i: d['error_rate'][i] if i in d['error_rate'].keys() else 0 for i in datakeys}
    duration = {i: d['duration'][i] if i in d['duration'].keys() else 0 for i in datakeys}
    return error_rate, duration


def sub_plot(data, alarm_start_minute, scores: list, subplot=(2, 2, 1), title='', test_length=60, train_length=240):
    data = dict(sorted(data.items(), key = lambda x: x[0]))
    x = data.keys()
    y = data.values()
    plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.plot(x, y)
    plt.vlines(alarm_start_minute, min(y), max(y), color='red', linestyles='--')
    plt.vlines(alarm_start_minute-test_length-train_length, min(y), max(y), color='b', linestyles='--')
    plt.vlines(alarm_start_minute-test_length, min(y), max(y), color='b', linestyles='--')
    plt.title(f'{title},score:{scores[subplot[2]-1]:.2f}', fontdict={'fontsize': 18})


def get_par_parent_nodes(G: nx.DiGraph, node_list):
    parent_nodes = set()
    for node in node_list:
        parent_nodes.update(G.predecessors(node))
    for node in node_list:
        if node in parent_nodes:
            parent_nodes.remove(node)
    return parent_nodes


def get_connected_graph(G: nx.DiGraph, nodes: List[str]) -> nx.DiGraph:
    Gout = nx.DiGraph()
    q = deque(nodes)
    visited = set()
    while q:
        start = q.popleft()
        for callee in G.successors(start):
            if f'{start}|{callee}' in visited:
                continue
            visited.add(f'{start}|{callee}')
            Gout.add_node(start, **G.nodes[start])
            Gout.add_node(callee, **G.nodes[callee])
            Gout.add_edge(start, callee, **G[start][callee])
            q.append(callee)
    q = deque(nodes)
    visited = set()
    while q:
        end = q.popleft()
        for caller in G.predecessors(end):
            if f'{caller}|{end}' in visited:
                continue
            visited.add(f'{caller}|{end}')
            Gout.add_node(caller, **G.nodes[caller])
            Gout.add_node(end, **G.nodes[end])
            Gout.add_edge(caller, end, **G[caller][end])
            q.append(caller)
    return Gout

def get_sorted_val(d: Dict, start: int, end: int) -> List:
    """
    Args:
        d: {0: v1, 1: v1, 3: v3, ...}
    Returns: 
        List: 在[start, end)区间内的key, 返回按key排序后对应的值的列表
    """
    kvs = {k: v for k, v in d.items() if k >= start and k < end}
    return [v for _, v in sorted(kvs.items(), key=lambda x: x[0])]


def parcorr(x: np.ndarray, y: np.ndarray, Z: np.ndarray = None) -> float:
    """ 计算x与y在Z条件下的偏相关系数, 当Z为None时, 退化为计算x与y的相关系数
    Args:
        x: (d, )   d为维度, 下同
        y: (d, )
        Z: (d, n)  n为x与y的parent变量的个数
    Returns:
        float: 偏相关系数
    """
    x_res = x
    y_res = y
    if Z is not None:
        x_beta = np.linalg.lstsq(Z, x, rcond=None)[0]
        y_beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        x_res = x - np.dot(Z, x_beta)
        y_res = y - np.dot(Z, y_beta)
    val, _ = pearsonr(x_res, y_res)
    return abs(val)


@timeit('Merging calling data ...')
def merge_callings_fro_to(callings: Dict, fro: str = 'method', to: str = 'server'):
    assert (to != fro)
    assert (fro !='server' and to !='method')
    calling_df = pd.DataFrame(callings.values())
    if len(calling_df) == 0:
        return {}

    # if fro == 'method':
    #     cols = ['caller_server', 'caller_service', 'caller_method',
    #             'callee_server', 'callee_service', 'callee_method']
    # elif fro == 'service':
    #     cols = ['caller_server', 'caller_service',
    #             'callee_server', 'callee_service']
    # else:
    #     raise ValueError(f'fro must be method or service, but got {fro}')

    # if to == 'server':
    #     gbcols = ['caller_server', 'callee_server']
    # elif to == 'service':
    #     gbcols = ['caller_server', 'caller_service', 'callee_server', 'callee_service']
    # else:
    #     raise ValueError(f'dest must be server or service, but got {to}')

    ## TODO: check
    cols = ['caller_server', 'caller_service',
            'callee_server', 'callee_service']
    gbcols = ['caller_server', 'callee_server']

    for i, col in enumerate(cols):
        calling_df[col] = calling_df['name'].apply(lambda x: x.split('|')[i])
    return merge_callings(calling_df, gbcols)


def split_calling(name):
    arr = name.split('|')
    caller = '|'.join(arr[:len(arr)//2])
    callee = '|'.join(arr[len(arr)//2:])
    return caller, callee


class GraphGen:
    def __init__(self, anomaly_detector, hyper_params, case: CaseModel):
        self.sim_start: int = case.alarm_start_minute - hyper_params['test_length_before'] + 1
        self.sim_end: int = case.alarm_start_minute + hyper_params['test_length_after']
        self.datakeys = list(range(self.sim_start - 1 - hyper_params['train_length'], self.sim_end + 1)) #list(range(1440))

        self.case = case
        self.anomaly_detector = anomaly_detector
        self.hyper_params = hyper_params

        self.method_node_graph: nx.DiGraph = None
        self.service_node_graph: nx.DiGraph = None
        self.server_node_graph: nx.DiGraph = None
        self.method_calling_node_graph: nx.DiGraph = None

        self.asso_method_node_ano_graph: nx.DiGraph = None
        self.asso_service_node_ano_graph: Tuple[nx.DiGraph, Dict] = None

        self.mix_method_node_graph: nx.DiGraph  = None
        self.mix_server_node_graph: nx.DiGraph = None

        self.method_score_df = None

    
    def aggr_data_list(self, data_list):
        aggr_data = {}
        keys = ['error_rate', 'duration']
        for k in keys:
            c = Counter({i: 0 for i in self.datakeys})
            for d in data_list:
                c.update(d[k])
            aggr_data[k] = dict(c)
        # FIXME: 字段
        duration = {i: aggr_data['duration'][i] for i in self.datakeys} 
        error_rate = {i: aggr_data['error_rate'][i] for i in self.datakeys}
        return {'duration': duration, 'error_rate': error_rate}

    
    def get_method_node_graph(self, require_data = False) -> nx.DiGraph:
        """
        return the entire graph on method level with anomaly score and similarity
        """
        if not self.method_node_graph:
            G = self._build_node_graph(self.case.callings, require_edge_data= require_data)
            self.method_node_graph = G
        return self.method_node_graph

    def get_service_node_graph(self, require_data = False) -> nx.DiGraph:
        if not self.service_node_graph:
            service_callings = merge_callings_fro_to(self.case.callings, fro='method', to='service')
            G = self._build_node_graph(service_callings, require_edge_data= require_data) 
            self.service_node_graph = G
        return self.service_node_graph
    
    def get_asso_method_node_ano_graph(self, require_edge_data = False) -> nx.DiGraph:
        """
        find the association graph on method level with anomaly degree.
        return the anomaly connected subgraph of the association graph on method level
        """
        if not self.asso_method_node_ano_graph:
            G = self._build_asso_node_graph(self.case.callings, require_edge_data=require_edge_data)
            G_ano = self._build_node_ano_graph(G)
            self.asso_method_node_ano_graph = G_ano
        return self.asso_method_node_ano_graph
    

    def get_asso_service_node_ano_graph(self, require_edge_data = False) -> nx.DiGraph:
        if not self.asso_service_node_ano_graph:
            callings = merge_callings_fro_to(self.case.callings, 'method', 'service')
            G = self._build_asso_node_graph(callings, require_edge_data=require_edge_data)
            G_ano = self._build_node_ano_graph(G)
            self.asso_service_node_ano_graph = (G_ano, callings)
        return self.asso_service_node_ano_graph


    def get_method_calling_node_graph(self, data_key='data', edge_weight_key='weight') -> nx.DiGraph:
        """
        return calling graph on method level
        """
        if self.method_calling_node_graph:
            return self.method_calling_node_graph
        G_ano = self.get_asso_method_node_ano_graph(require_edge_data = True)
        Gout = self._build_calling_node_graph(G_ano, node_data_key=data_key, edge_weight_key=edge_weight_key)
        self.method_calling_node_graph = Gout
        return Gout
    
    
    @timeit('**********Building server node graph**********')
    def get_server_node_graph(self, node_data_key='data', edge_weight_key='weight') -> nx.DiGraph:
        if self.server_node_graph:
            return self.server_node_graph        

        if self.hyper_params['level'] == 'service':
            Gano, service_callings = self.get_asso_service_node_ano_graph()
            callings = {f'{fro}|{to}': service_callings[f'{fro}|{to}'] for fro, to in Gano.edges}
            merged_callings = merge_callings_fro_to(callings, 'service', 'server')

        elif self.hyper_params['level'] == 'method':
            Gano = self.get_asso_method_node_ano_graph()
            callings = {f'{fro}|{to}': self.case.callings[f'{fro}|{to}'] for fro, to in Gano.edges}
            merged_callings = merge_callings_fro_to(callings, 'method', 'server')

        as_callee = defaultdict(list)
        as_caller = defaultdict(list)
        Gout = nx.DiGraph()
        for name, d in merged_callings.items():
            caller, callee = split_calling(name)
            as_callee[callee].append(d)
            as_caller[caller].append(d)
            Gout.add_edge(caller, callee)

        # 对于图中的每个点，将该点作为被调和主调时的数据聚合，并计算失败率的时间序列数据作为该节点的数据
        for node in set(as_callee) | set(as_caller):
            node_data = self.aggr_data_list(as_callee[node] + as_caller[node])
            Gout.nodes[node][node_data_key] = node_data


        # for callee, data_list in as_callee.items():
        #     err_rate = self.aggr_data_list(data_list)
        #     Gout.nodes[callee][node_data_key] = err_rate
        # # 移除排序图中 没有被调用的节点
        # for node in [n for n in Gout.nodes if n not in as_callee]:
        #     Gout.remove_node(node)
 
        for fro, to in Gout.edges:
            fro_data = Gout.nodes[fro][node_data_key]
            to_data = Gout.nodes[to][node_data_key]
            weight = 0
            for k in ['duration', 'error_rate']:
                x = get_sorted_val(fro_data[k], self.sim_start, self.sim_end)
                y = get_sorted_val(to_data[k], self.sim_start, self.sim_end)
                if np.std(x) != 0 and np.std(y) != 0:
                    weight = max(weight, abs(pearsonr(x, y)[0]))
            Gout[fro][to][edge_weight_key] = weight

        self.server_node_graph = Gout
        return Gout
        

    def get_mix_method_node_graph(self) -> nx.DiGraph:
        """
        return mixed graph on method level
        """
        if self.mix_method_node_graph:
            return self.mix_method_node_graph
        Gano = self.get_asso_method_node_ano_graph()
        callings = {}
        for fro, to in Gano.edges():
            name = f'{fro}|{to}'
            callings[name] = self.case.callings[name]

        Gmix = self._build_mix_node_graph(callings)
        self.mix_method_node_graph = Gmix
        return Gmix
    
    
    def get_mix_server_node_graph(self) -> nx.DiGraph:
        """
        return mixed graph on server level merged by service or method
        """
        if self.mix_server_node_graph:
            return self.mix_server_node_graph

        level = self.hyper_params['level']
        base_callings = self.case.callings
        if level == 'service':
            Gano, base_callings = self.get_asso_service_node_ano_graph()
        elif level == 'method':
            Gano = self.get_asso_method_node_ano_graph()
        else:
            raise ValueError(f'level {level} is not supported')

        callings = {}
        for fro, to in Gano.edges():
            name = f'{fro}|{to}'
            callings[name] = base_callings[name]
        
        merged_callings = merge_callings_fro_to(callings, level, 'server')
        self.mix_server_node_graph = self._build_mix_node_graph(merged_callings)
        return self.mix_server_node_graph

    
    def get_mix_server_node_graph_womethod(self) -> nx.DiGraph:
        """
        return the mixed graph without the detection on method level
        """
        if self.mix_server_node_graph:
            return self.mix_server_node_graph
        # server_callings = merge_callings_for_server(self.case.callings)
        server_callings = merge_callings_fro_to(self.case.callings, 'method', 'server')
        Gasso = self._build_asso_node_graph(server_callings)

        Gano = self._build_node_ano_graph(Gasso)

        callings = {}
        for fro, to in Gano.edges():
            name = f'{fro}|{to}'
            callings[name] = server_callings[name]

        self.mix_server_node_graph = self._build_mix_node_graph(callings)
        return self.mix_server_node_graph 


    # new
    @timeit('**********Building association graph**********')
    def _build_asso_node_graph(self,
            callings,
            edge_score_key='score', 
            edge_similarity_key='similarity', 
            edge_data_key='data',
            require_edge_data=False) -> nx.DiGraph:

        case = self.case
        G_total = nx.DiGraph()
        for name, _ in callings.items():
            caller, callee = split_calling(name)
            G_total.add_edge(caller, callee)

        start_server = self.case.alarm_item
        qnodes = [k for k in G_total.nodes if start_server in k]

        ## 获得与qnodes连通的子图
        G = get_connected_graph(G_total, qnodes)
        
        ano_dtr = self.anomaly_detector
        for fro, to in G.edges:
            name = f'{fro}|{to}'
            d = callings[name]

            err_rate, duration = cal_data_rate(d, self.datakeys)
            is_alarm = False
            if case.alarm_item in name:
                is_alarm = True
            scores = ano_dtr.edge_score(case.alarm_start_minute, [err_rate, duration], is_alarm)
            # scores = ano_dtr.edge_score(case.alarm_start_minute, [except_rate, timeout_rate, d['exception_min'], d['timeout_min']])


            edge_attrs = {edge_score_key: max(scores)}
            if require_edge_data:
                # FIXME: 边的数据
                duration_vals = get_sorted_val(duration, self.sim_start, self.sim_end)
                error_vals = get_sorted_val(err_rate, self.sim_start, self.sim_end)
                edge_attrs[edge_data_key] = {'duration': np.array(duration_vals), 'error': np.array(error_vals)}
            G.add_edge(fro, to, **edge_attrs)
            
        return G
        

    @timeit('**********Building node graph**********')
    def _build_node_graph(self, callings: Dict, 
            edge_score_key='score', 
            edge_data_key='data',
            require_edge_data=False):
        """ 构造以method/server为node的图, 返回图中的边有异常分数、相似度（、数据）
        Args:
            callings:  self.case.callings |  merge后的server callings
        Returns:
            nx.DiGraph: 以method/server为节点的图
            The output level is depend on the input callings.
        """
        case = self.case
        anomaly_detector = self.anomaly_detector
        G = nx.DiGraph()
        for name, d in callings.items():
            caller, callee = split_calling(name)

            error_rate, duration = cal_data_rate(d, self.datakeys)
            is_alarm = False
            if case.alarm_item in name:
                is_alarm = True
            scores = anomaly_detector.edge_score(case.alarm_start_minute, [error_rate, duration], is_alarm)
            
            ## 设置边的属性
            edge_attrs = {edge_score_key: max(scores)}
            if require_edge_data:
                # FIXME: 边的数据
                duration_vals = get_sorted_val(duration, self.sim_start, self.sim_end)
                error_vals = get_sorted_val(error_rate, self.sim_start, self.sim_end)
                edge_attrs[edge_data_key] = {'duration': np.array(duration_vals), 'error_rate': np.array(error_vals)}
            G.add_edge(caller, callee, **edge_attrs)
        return G

    
    @timeit('**********Building anomaly node graph**********')
    def _build_node_ano_graph(self, G: nx.DiGraph, ano_score_key='score'):
        """ 构造以method/server为节点的异常子图
        Args:
            G:  method/server为节点的图 (self._build_node_graph()的返回值)
        Returns:
            G_ano:  method/server为节点的异常子图
        """
        G_ano = nx.DiGraph()
        for fro, to, data in G.edges(data=True):
            if data[ano_score_key] != 0:
                G_ano.add_edge(fro, to, **data)

        start_server = self.case.alarm_item
        qnodes = [k for k in G_ano.nodes if start_server in k]
        return get_connected_graph(G_ano, qnodes)


    @timeit('**********Building calling node graph**********')
    def _build_calling_node_graph(self, G: nx.DiGraph, corr_method='pearson', node_data_key='data', edge_weight_key='weight') -> nx.DiGraph:
        """ 从以method/server为节点的图`G`构建以调用边为节点的图

        Parameters
        ----------
        corr_method : str, optional
            `pearson` | `par_corr`, 边权重的算法
        node_data_key : str, optional
            与调用`_build_node_graph`时的`edge_data_key`一样

        Returns
        -------
        nx.DiGraph
            以调用边为节点的图
        """
        # 构造无权重的calling node graph
        Gout = self._build_calling_node_graph_skeleton(G)
        # 给calling node graph边加权重
        for start, end in Gout.edges():
            x_data = Gout.nodes[start][node_data_key]
            y_data = Gout.nodes[end][node_data_key]
            corr_val = 0
            if corr_method == 'pearson':
                for k in ['duration', 'error_rate']:
                    x = x_data[k]
                    y = y_data[k]
                    if np.std(x) != 0 and np.std(y) != 0:
                        v, _ = pearsonr(x, y)
                        corr_val = max(corr_val, abs(v))
            # elif corr_method == 'par_corr':
            #     parent_nodes = get_par_parent_nodes(Gout, [start, end])
            #     z_data = [Gout.nodes[node][node_data_key] for node in parent_nodes]
            #     z_data = np.array(z_data).reshape(-1, len(z_data)) if len(z_data) else None
            #     corr_val = parcorr(x_data, y_data, z_data)
            else:
                raise ValueError(f'corr_method: {corr_method} is not implemented')
            Gout[start][end][edge_weight_key] = corr_val
        return Gout


    def _build_calling_node_graph_skeleton(self, G: nx.DiGraph) -> nx.DiGraph:
        """ 从以method/server为节点的图`G`构建以调用边为节点的图

        Returns
        -------
        nx.DiGraph
            以调用边为节点的图, 图上的顶点和边都没有数据
        """
        start_server = self.case.alarm_item
        qnodes = [n for n in G.nodes if start_server in n]

        Gout = nx.DiGraph()
        # 向上寻找
        callings_set = set()
        for end in qnodes:
            for caller in G.predecessors(end):
                callings_set.add(f'{caller}|{end}')

        callings_q = deque(callings_set)
        visited = set()
        while callings_q:
            end_calling = callings_q.popleft()
            mid, end = split_calling(end_calling)

            Gout.add_node(end_calling, **G[mid][end]) 

            for caller in G.predecessors(mid):
                start_calling = f'{caller}|{mid}'
                if f'{start_calling}|{end_calling}' in visited:
                    continue
                visited.add(f'{start_calling}|{end_calling}')
                Gout.add_node(start_calling, **G[caller][mid])
                Gout.add_edge(start_calling, end_calling)
                callings_q.append(start_calling)

        if not callings_set:
            for start in qnodes:
                for callee in G.successors(start):
                    callings_set.add(f'{start}|{callee}')

        # 向下寻找
        callings_q = deque(callings_set)
        visited = set()
        while callings_q:
            start_calling = callings_q.popleft()
            start, mid = split_calling(start_calling)

            Gout.add_node(start_calling, **G[start][mid])  # add

            for callee in G.successors(mid):
                end_calling = f'{mid}|{callee}'
                if f'{start_calling}|{end_calling}' in visited:
                    continue
                visited.add(f'{start_calling}|{end_calling}')
                Gout.add_node(end_calling, **G[mid][callee])
                Gout.add_edge(start_calling, end_calling)
                callings_q.append(end_calling)

        return Gout



    @timeit('**********Building mix node graph**********')
    def _build_mix_node_graph(self, callings, node_data_key='data', edge_weight_key='weight') -> nx.DiGraph:
        """ 根据调用构建混合图
        Args:
            callings: method/service/server层的调用{k:v}
        """
        ## 0. 构造以method/service/server为节点的图 
        G = self._build_node_graph(callings, require_edge_data=True)
        ## 1. 构造以calling为节点的图
        Gmix = self._build_calling_node_graph(G, corr_method='pearson', 
            node_data_key=node_data_key, edge_weight_key=edge_weight_key)

        as_callee = defaultdict(list)
        as_caller = defaultdict(list)
        for name, d in callings.items():
            caller, callee = split_calling(name)
            as_callee[callee].append(d)
            as_caller[caller].append(d)
        

        ## 2. 在calling图上添加混合边与节点
        calling_nodes = list(Gmix.nodes)
        for node in calling_nodes:
            caller, callee = split_calling(node)
            # server node: item
            if caller not in Gmix.nodes():
                caller_data = self.aggr_data_list(as_caller[caller] + as_callee[caller])
                Gmix.add_node(caller, type='item', data = caller_data)
            if callee not in Gmix.nodes():
                callee_data = self.aggr_data_list(as_caller[callee] + as_callee[callee])
                Gmix.add_node(callee, type='item', data = callee_data)
            # calling -> server: mix edge
            Gmix.add_edge(node, caller, type='mix')
            Gmix.add_edge(node, callee, type='mix')

        ## 3. 计算所有混合边的权重
        server_score = {}
        new_col = self.hyper_params['new_col']
        beta = self.hyper_params['beta']
        if beta != 0.0:
            if self.method_score_df is not None and len(self.method_score_df) != 0:
                for i, g in self.method_score_df.groupby('server'):
                    server_score[i] = max(g[new_col], default=0)
            else:
                print('method_score_df is None or len(method_score_df) is zero')
        
        for node in Gmix:
            if Gmix.nodes[node].get('type') == 'item':
                continue
            caller, callee = split_calling(node)

            minw  = max((data_[edge_weight_key] for (_, _, data_) in Gmix.in_edges(node, data=True)), default=0)
            moutw = max((data_[edge_weight_key] for (_, _, data_) in Gmix.out_edges(node, data=True) if data_.get('type') != 'mix'), default=0)
            
            total_w = max(0, minw - moutw)
            if minw == 0 and moutw == 0: total_w = 1
            elif minw == 0: total_w = 1 - moutw
            elif moutw == 0: total_w = minw

            if self.hyper_params['server_data']:
                calling_data = Gmix.nodes[node][node_data_key]
                caller_data = Gmix.nodes[caller]['data']
                callee_data = Gmix.nodes[callee]['data']
                caller_corr = 0
                callee_corr = 0
                for k in ['error_rate', 'duration']:
                    x = get_sorted_val(calling_data[k], self.sim_start, self.sim_end)
                    y1 = get_sorted_val(caller_data[k], self.sim_start, self.sim_end)
                    y2 = get_sorted_val(callee_data[k], self.sim_start, self.sim_end)
                    if np.std(x) != 0 and np.std(y1) != 0:
                        caller_corr = max(caller_corr, abs(pearsonr(x, y1)[0]))
                    if np.std(x) != 0 and np.std(y2) != 0:
                        callee_corr = max(callee_corr, abs(pearsonr(x, y2)[0]))

                if caller_corr == 0 and callee_corr == 0:
                    Gmix[node][caller][edge_weight_key] = total_w * 0.5
                    Gmix[node][callee][edge_weight_key] = total_w * 0.5
                else:
                    Gmix[node][caller][edge_weight_key] = total_w * caller_corr / (caller_corr + callee_corr)
                    Gmix[node][callee][edge_weight_key] = total_w * callee_corr / (caller_corr + callee_corr)
            else:
                if beta != 0.0:
                    caller_score = server_score.get(caller, 0)
                    callee_score = server_score.get(callee, 0)
                    caller_w, callee_w = (0.5 + beta, 0.5 - beta) if caller_score > callee_score else (0.5 - beta, 0.5 + beta)
                else:
                    caller_w, callee_w = 0.5, 0.5
                
                Gmix[node][caller][edge_weight_key] = total_w * caller_w
                Gmix[node][callee][edge_weight_key] = total_w * callee_w


        return Gmix


    def draw_related_callings(self, callings, G_ano, figure_out):
        ## TODO: fix
        case = self.case
        hyper_params = self.hyper_params

        related_nodes = []
        for c in nx.connected_components(G_ano.to_undirected()):
            if case.alarm_item in '|'.join(c):
                related_nodes.extend(c)

        calls_num = 0
        anomaly_callings = {'|'.join(i) for i in G_ano.edges}
        Gout = Digraph(format='pdf')
        draw_nodes = []
        for name, d in callings.items():
            if name not in anomaly_callings:
                continue
            if all(i not in name for i in related_nodes):
                continue

            scores = np.zeros(5)

            plt.figure(figsize=(15, 8))
            fro, to = split_calling(name)
            plt.suptitle(f'{fro}\n|{to}', fontsize=15)
            
            
            sub_plot({k:v for k,v in d['error_rate'].items() if (k > case.alarm_start_minute - hyper_params['test_length_before'] - hyper_params['train_length']- 10) and (k<case.alarm_start_minute +hyper_params['test_length_after']+10)}, 
                     case.alarm_start_minute, scores, (2, 1, 1),
                     title='err_rate', test_length=hyper_params['test_length_before'], train_length=hyper_params['train_length'])
            sub_plot({k:v for k,v in d['duration'].items() if (k > case.alarm_start_minute - hyper_params['test_length_before'] - hyper_params['train_length']- 10) and (k<case.alarm_start_minute +hyper_params['test_length_after']+10)}, 
                     case.alarm_start_minute, scores, (2, 1, 2),
                     title='duration', test_length=hyper_params['test_length_before'], train_length=hyper_params['train_length'])

            plt.savefig(os.path.join(
                figure_out, f"{calls_num}_{fro.split('|')[0]}_{to.split('|')[0]}.png"))
            plt.close()
            calls_num += 1


            if case.rc[0] in name:
                Gout.edge(fro, to, color='green')
            else:
                Gout.edge(fro, to)
            draw_nodes.extend((fro, to))

        for k in draw_nodes:
            if case.alarm_item in k:
                Gout.node(k, label=f'{k}', color='red')
            elif case.rc[0] in k:
                Gout.node(k, label=f'{k}', color='green')
            else:
                Gout.node(k, label=f'{k}')

        Gout.render(os.path.join(figure_out, 'Callings'), cleanup=True)

        
    def draw_mix_graph(self, figure_out, edge_weight_key='weight'):
        case = self.case
        Gmix = self.get_mix_server_node_graph()
        dot = Digraph(format='pdf')
        for node in Gmix.nodes:
            attrs = {}
            if Gmix.nodes[node].get('type') != 'item':
                attrs['shape'] = 'box'
            if case.alarm_item in node:
                attrs['color'] = 'red'
            elif case.rc[0] in node:
                attrs['color'] = 'green'

            dot.node(node, **attrs, label = f'{node}-{round(Gmix.nodes[node].get("score", 0), 2)}')

        for fro, to in Gmix.edges:
            weight = Gmix[fro][to][edge_weight_key]
            dot.edge(fro, to, label=f'{weight:.4f}')

        dot.render(os.path.join(figure_out, 'MixGraph'), cleanup=True)


    def draw_server_node_graph(self, figure_out, edge_weight_key='weight'):
        G = self.get_server_node_graph()
        case = self.case
        Gout = Digraph(format='pdf')
        for nod in G.nodes():
            if nod == case.alarm_item:
                Gout.node(nod, label=f'{nod}', color='red')
            if nod in case.rc:
                Gout.node(nod, label=f'{nod}', color='green')
            else:
                Gout.node(nod, label=f'{nod}')
        for fro, to in G.edges():
            weight = G[fro][to][edge_weight_key]
            if fro in case.rc or to in case.rc:
                Gout.edge(fro, to, color = 'green', label=f'{weight:.4f}')
            else:
                Gout.edge(fro, to, label=f'{weight:.4f}')
        Gout.render(os.path.join(figure_out, 'ServerNodes'), cleanup=True)
