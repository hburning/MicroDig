import sys
from graph import GraphGen, get_connected_graph
from collections import defaultdict, deque
import networkx as nx
import numpy as np
import pandas as pd
from typing import List

def pagerank(G: nx.DiGraph, weight='weight'):
    try:
        pr = nx.pagerank(G, weight=weight)
    except nx.PowerIterationFailedConvergence:
        try:
            pr = nx.pagerank_numpy(G, weight=weight)
        except nx.PowerIterationFailedConvergence:
            print('PAGERANK FAILED, all nodes have equal score', file=sys.stderr)
            n_nodes = len(G.nodes)
            pr = {n: 1 / n_nodes for n in G.nodes}
    return pr


def random_walk(G: nx.DiGraph, start_nodes, weight='weight', iters=1000):
    v_s = np.zeros(len(G))
    if len(start_nodes) == 0:
        return dict(zip(G, v_s))

    for i, n in enumerate(G):
        if n in start_nodes.keys():
            v_s[i] = start_nodes[n]
    v_s = v_s / np.sum(v_s)

    p_mat = nx.to_numpy_array(G, weight=weight)
    p_mat = p_mat / (np.sum(p_mat, axis=1, keepdims=True)) # 要求每行都不是全0

    # set seed for numpy
    np.random.seed(2022)
    # steps = len(G.nodes())
    steps = 50
    cnt = [0] * len(G)
    for _ in range(iters):
        node = np.random.choice(range(len(G)), p=v_s)
        for _ in range(steps):
            try:
                node = np.random.choice(range(len(G)), p=p_mat[node])
                cnt[node] += 1
            except Exception:
                break
    v_s = np.array(cnt) / np.sum(cnt)

    return dict(zip(G, v_s))


def last2_anomaly_hop_search(G_ano: nx.DiGraph, start_server: str, sim_thre=0.7, sim_key='similarity'):
    q = deque(n for n in G_ano.nodes if start_server in n)
    remain_nodes = set()
    end_nodes = set()
    while q:
        start = q.popleft()
        callees = list(G_ano.successors(start))
        if not callees:
            end_nodes.add(start)
            continue
        for callee in callees:
            edge_data = G_ano[start][callee]
            if edge_data[sim_key] > sim_thre:
                q.append(callee)
                remain_nodes.add(start)
                remain_nodes.add(callee)
    res = end_nodes.copy()
    for end_node in end_nodes:
        for node in G_ano.predecessors(end_node):
            if node in remain_nodes:
                res.add(node)
    return list(res)


# def anomaly_hop_search(G_ano: nx.DiGraph, start_server: str) -> List:
#     # 找到异常子图中的所有包含start_server的节点
#     q = [n for n in G_ano.nodes if start_server in n]
#     G_asso = get_connected_graph(G_ano, q)
#     return list(G_asso)

def deepest2_anomaly_hop_search(G_ano: nx.DiGraph, start_server: str, sim_thre=0.7, sim_key='similarity') -> List:
    # 找到异常子图中的所有包含start_server的节点
    q = deque(n for n in G_ano.nodes if start_server in n)
    remain_nodes = {n: 0 for n in q}
    # 以这些节点为初始节点，进行广度优先搜索
    while q:
        start = q.popleft()
        start_depth = remain_nodes[start]
        for callee in G_ano.successors(start):
            edge_data = G_ano[start][callee]
            if edge_data[sim_key] >= sim_thre:
                remain_nodes[callee] = start_depth + 1
                q.append(callee)
    max_depth = max(remain_nodes.values()) if len(remain_nodes) else 0
    return [k for k, v in remain_nodes.items() if v >= max_depth - 1]

def calling_node_pr(G: nx.DiGraph, edge_weight_key='weight'):
    pr = pagerank(G, weight=edge_weight_key)
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)



class Ranker:
    def __init__(self, hyper_params, case, graph_gen: GraphGen):
        self.hyper_params = hyper_params
        self.case = case
        self.graph_gen: GraphGen = graph_gen
        self.candidate_root_causes = None

    # 找到所有异常的method节点
    def get_candidate_root_causes(self):
        if self.candidate_root_causes:
            return self.candidate_root_causes
        case = self.case
        start_server = case.alarm_item
        level = self.hyper_params['level']
        # 1. 获得异常子图
        if level == 'method': 
            G = self.graph_gen.get_asso_method_node_ano_graph()
        elif level == 'service':
            G, _ = self.graph_gen.get_asso_service_node_ano_graph()
        else:
            raise ValueError(f'level {level} is not supported')

        # 2. 设置相似度阈值并搜索候选根因
        # sim_thre = -1 if case.changed else -1
        sim_thre = -1
        candidate_root_causes = []
        search_method = self.hyper_params['search_method']
        if search_method == 'all':
            candidate_root_causes = list(G.nodes)
        elif search_method == 'deepest2':
            candidate_root_causes = deepest2_anomaly_hop_search(G, start_server, sim_thre=sim_thre)
        elif search_method == 'last2':
            candidate_root_causes = last2_anomaly_hop_search(G, start_server, sim_thre=sim_thre)
        else:
            raise ValueError(f'Unknown search method: {search_method}')
        self.candidate_root_causes = candidate_root_causes
        return candidate_root_causes

    def server_node_rank(self):
        Gserver = self.graph_gen.get_server_node_graph()
        if len(Gserver) == 0:
            return pd.DataFrame()
        res = self.rank_server_node_graph(Gserver)
        return pd.DataFrame(res, columns=['name', 'final_score'])


    def mix_node_rank_wodetect(self):
        Gmix = self.graph_gen.get_mix_server_node_graph_womethod()
        if len(Gmix) == 0:
            return pd.DataFrame()
        res = self.rank_mix_graph(Gmix)
        return pd.DataFrame(res, columns=['name', 'final_score'])


    def mix_node_rank(self):
        Gmix = self.graph_gen.get_mix_server_node_graph()
        if len(Gmix) == 0:
            return pd.DataFrame()
        res = self.rank_mix_graph(Gmix)
        return pd.DataFrame(res, columns=['name', 'final_score'])


    # def alg3(self):
    #     Gmix = self.graph_gen.get_mix_method_node_graph()
    #     return self.rank_mix_graph(Gmix)
        

    # def alg2(self):
    #     G_calling = self.graph_gen.get_method_calling_node_graph(data_key='data')
    #     return calling_node_pr(G_calling)


    def rank_server_node_graph(self, G: nx.DiGraph, edge_weight_key='weight'):
        arr = nx.to_numpy_array(G, weight=edge_weight_key)
        # 反向边
        rev_mat = arr.T * (arr == 0) * self.hyper_params['rev_weight']
        # 自环
        in_w = np.amax(arr, axis=0, keepdims=True)
        out_w = np.amax(arr, axis=1, keepdims=True)
        self_mat = np.diag(np.diag(np.maximum(0, in_w - out_w)))
        arr += rev_mat + self_mat
        arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-8)
        Garr = nx.from_numpy_array(arr, create_using=nx.DiGraph)
        Garr = nx.relabel_nodes(Garr, dict(zip(range(arr.shape[0]), list(G.nodes))))
        pr = self.do_rank(Garr, edge_weight_key)
        return list(sorted(pr.items(), key=lambda x: x[1], reverse=True))


    def rank_mix_graph(self, Gmix: nx.DiGraph, edge_weight_key='weight'):
        arr = nx.to_numpy_array(Gmix, weight=edge_weight_key)

        # 反向边
        rank_method = self.hyper_params['rank_method']
        if rank_method == 'pagerank':
            anno_key = 'anno'
            for fro, to, t in Gmix.edges.data('type', default=''):
                if t != 'mix':
                    Gmix[fro][to][anno_key] = 2

            arr_helper = nx.to_numpy_array(Gmix, weight=anno_key)
            arr_helper = (arr_helper == 2)     # 所非mix边的mask
            arr += arr.T * (arr == 0) * self.hyper_params['rev_weight'] * arr_helper.T
        elif rank_method == 'random walk':
            arr += arr.T * (arr == 0) * self.hyper_params['rev_weight']
            # server 节点自环
            server_node_idxs = [i for i, node in enumerate(Gmix)]
            for i in server_node_idxs:
                # arr[i, i] = np.sum(arr[:, i]) * (1-self.hyper_params['rev_weight'])
                # arr[i, i] = 1
                arr[i, i] = np.max(arr[:, i])
        else:
            raise ValueError(f'Unknown rank method: {rank_method}')

        # 出边归一化
        arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-8)
        Garr = nx.from_numpy_array(arr, create_using=nx.DiGraph)
        Garr = nx.relabel_nodes(Garr, dict(zip(range(arr.shape[0]), list(Gmix.nodes))))
        for node in Garr.nodes():
            Garr.add_node(node, **Gmix.nodes[node])
        Gmix = Garr

        # rank
        pr = self.do_rank(Gmix, edge_weight_key, graph_type='mix')
        sorted_items = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return [item for item in sorted_items if Gmix.nodes[item[0]].get('type', '') == 'item']


    def do_rank(self, G: nx.DiGraph, weight, graph_type='normal'):
        rank_method = self.hyper_params['rank_method']
        if rank_method == 'random walk':
            if graph_type == 'normal':
                start_nodes = dict([(node,1) for node in G if self.case.alarm_item in node])
            elif graph_type == 'mix':
                candidate_nodes = [(node, G.nodes[node]['score']) for node in G \
                    if self.case.alarm_item in node and G.nodes[node].get('type', '') != 'item']
                # candidate_nodes = [(node, G.nodes[node]['score']) for node in G \
                #     if G.nodes[node].get('type', '') != 'item']
                start_nodes = dict(sorted(candidate_nodes, key=lambda x : x[1], reverse=True))
            # TODO: bug, start_nodes may be empty
            # print(f'start_nodes = {start_nodes}')
            pr = random_walk(G, start_nodes, weight=weight, iters=100)
        elif rank_method == 'pagerank':
            pr = pagerank(G, weight=weight)
        else:
            raise ValueError(f'Unknown rank method: {rank_method}')
        return pr


    def alg1(self, edge_s_w = 0.5, num_ratio_w = 0.5, method_s_w=0.5, server_s_w=0.5, edge_score_key='score'):
        level = self.hyper_params['level']
        method_score_df = self._cal_method_node_scores(edge_s_w, num_ratio_w, edge_score_key=edge_score_key)
        if not method_score_df.shape[0]:
            return pd.DataFrame(), pd.DataFrame()
        server_score_df = self._cal_server_node_scores(edge_s_w, num_ratio_w, edge_score_key=edge_score_key)
        mean_m = method_score_df[f'{level}_score'].mean()
        std_m = method_score_df[f'{level}_score'].std()
        mean_s = server_score_df['server_score'].mean()
        std_s = server_score_df['server_score'].std()

        # method和server分数分别进行加权标准化
        method_score_df[f'{level}_score_norm'] = method_s_w * (method_score_df[f'{level}_score'] - mean_m) / std_m if std_m > 1e-6 else method_s_w
        server_score_df['server_score_norm'] = server_s_w * (server_score_df['server_score'] - mean_s) / std_s if std_s > 1e-6 else server_s_w

        # 合并，计算method分数
        method_score_df = method_score_df.merge(server_score_df, on='server')
        method_score_df['final_score'] = method_score_df[f'{level}_score_norm'] + method_score_df['server_score_norm']

        # 分数排序
        server_score_df = server_score_df.sort_values(by='server_score_norm', ascending=False)
        server_score_df.rename(columns = {'server': 'name'}, inplace=True)
        method_score_df = method_score_df.sort_values(by='final_score', ascending=False)
        return server_score_df, method_score_df


    def _cal_method_node_scores(self, edge_s_w, num_ratio_w, edge_score_key='score'):
        ek = edge_score_key
        level = self.hyper_params['level']
        if level == 'method':
            G = self.graph_gen.get_method_node_graph()
        elif level == 'service':
            G = self.graph_gen.get_service_node_graph()
        candidate_nodes = self.get_candidate_root_causes()
        node_scores = {n: {} for n in candidate_nodes}
        for node in candidate_nodes:
            ## 1. 边的异常分数加权来得到节点的一个异常分数
            ano_caller_s = [G[pred][node][ek] for pred in G.pred[node] if G[pred][node][ek] > 0]
            ano_callee_s = [G[node][succ][ek] for succ in G.succ[node] if G[node][succ][ek] > 0]
            if ano_caller_s and ano_callee_s:
                score = (np.mean(ano_callee_s) + np.mean(ano_caller_s)) / 2
            else:
                score = np.mean(ano_callee_s + ano_caller_s)
            ano_edge_s = score

            ## 2. 节点被调、主调的异常调用占比两者相乘 
            caller_s = len(ano_caller_s) / len(G.pred[node]) if len(G.pred[node]) else 1
            callee_s = len(ano_callee_s) / len(G.succ[node]) if len(G.succ[node]) else 1
            # ano_num_ratio_s = caller_s * callee_s
            ano_num_ratio_s = (len(ano_caller_s) + len(ano_callee_s)) / (len(G.succ[node]) + len(G.pred[node])) if len(G.succ[node]) or len(G.pred[node]) else 1

            ## 3. 1和2加权得到节点分数
            node_scores[node][f'{level}'] = node
            node_scores[node]['server'] = node.split('|')[0]
            node_scores[node][f'{level}_score'] = ano_edge_s * edge_s_w + ano_num_ratio_s * num_ratio_w
            node_scores[node][f'{level}_anomaly_score'] = ano_edge_s
            node_scores[node][f'{level}_num_ratio_score'] = ano_num_ratio_s
            node_scores[node][f'{level}_anomaly_caller_num'] = len(ano_caller_s)
            node_scores[node][f'{level}_anomaly_caller_ratio'] = caller_s
            node_scores[node][f'{level}_anomaly_callee_num'] = len(ano_callee_s)
            node_scores[node][f'{level}_anomaly_callee_ratio'] = callee_s

        return pd.DataFrame(node_scores.values())


    def _cal_server_node_scores(self, edge_s_w, num_ratio_w, edge_score_key='score'):
        level = self.hyper_params['level']
        if level == 'method':
            G = self.graph_gen.get_method_node_graph()
        elif level == 'service':
            G = self.graph_gen.get_service_node_graph()
        candidate_nodes = self.get_candidate_root_causes()
        server_nodes = {n.split('|')[0] for n in candidate_nodes}
        server2method = {s: set() for s in server_nodes}
        for method in G.nodes:
            s = method.split('|')[0]
            if s in server2method:
                server2method[s].add(method)

        server_scores = {s: {} for s in server_nodes}
        for server in server_nodes:
            caller_scores = defaultdict(list)
            callee_scores = defaultdict(list)
            for method in server2method[server]:
                # 找到server作为被调时，所有调边，并计算它与各调用它的server之间的分数(包括异常和不异常)
                for caller in G.predecessors(method):
                    caller_server = caller.split('|')[0]
                    caller_scores[caller_server].append(G[caller][method][edge_score_key])

                # server 作为主调
                for callee in G.successors(method):
                    callee_server = callee.split('|')[0]
                    callee_scores[callee_server].append(G[method][callee][edge_score_key])

            caller_server_s = {k: np.mean(v) for k, v in caller_scores.items()}
            callee_server_s = {k: np.mean(v) for k, v in callee_scores.items()}
                                                                                                                                           
            ## 3. 所有candidate节点的server，算异常分数：通过内部所有被调和主调的method的异常分数加和
            ano_caller_s = [v for v in caller_server_s.values() if v > 0]
            ano_callee_s = [v for v in callee_server_s.values() if v > 0]
            ano_score = np.mean(ano_caller_s + ano_callee_s)

            ## 4. server的 被调、主调的异常调用数量和占比（server级别） 四者相乘
            caller_s = (len(ano_caller_s) / len(caller_server_s)) if len(caller_server_s) else 1
            callee_s = (len(ano_callee_s) / len(callee_server_s)) if len(callee_server_s) else 1
            # ano_num_ratio_s = caller_s * callee_s
            ano_num_ratio_s = (len(ano_caller_s) + len(ano_callee_s)) / (len(caller_server_s) + len(callee_server_s)) if len(caller_server_s) or len(callee_server_s) else 1
            

            server_scores[server]['server'] = server
            server_scores[server]['server_score'] = ano_score * edge_s_w + ano_num_ratio_s * num_ratio_w
            server_scores[server]['server_anomaly_score'] = ano_score
            server_scores[server]['server_num_ratio_score'] = ano_num_ratio_s
            server_scores[server]['server_anomaly_caller_num'] = len(ano_caller_s)
            server_scores[server]['server_anomaly_caller_ratio'] = caller_s
            server_scores[server]['server_anomaly_callee_num'] = len(ano_callee_s)
            server_scores[server]['server_anomaly_callee_ratio'] = callee_s

        return pd.DataFrame(server_scores.values())

