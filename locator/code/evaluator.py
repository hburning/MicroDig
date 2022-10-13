import os
import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, result_dir):
        self.result_mar = []
        self.result_mar_method = []
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.tmp_output = open(os.path.join(result_dir, 'tmp_result.csv'), 'w')
        self.tmp_output.write('alarm_start_time,mar\n', )
        self.tmp_output.flush()

    def save_tmp_mar(self, case, mar):
        self.tmp_output.write(f'{case.alarm_start_time},{mar}\n')
        self.tmp_output.flush()

    def process_result_method(self, res_df, case, score_col = 'final_score'):
        adjust = - min(res_df[score_col])
        res_df['final_score_adjusted'] = res_df[score_col] + adjust
        res_df.sort_values('final_score_adjusted', ascending = False).to_csv(os.path.join(self.result_dir, f'method-{case.alarm_start_time}.csv'), index=False)

        # res_df['server'] = res_df.apply(lambda x: x['nodes'].split('|')[0], axis=1)
        groups = res_df.groupby('server')
        result_servers = {i: sum(g['final_score_adjusted']) for i, g in groups}
        result_servers = list(dict(sorted(result_servers.items(), key=lambda x: x[1], reverse=True)).keys())
        mar = [result_servers.index(i)+1 if i in result_servers else len(result_servers)+(case.server_num-len(result_servers))/2 for i in case.rc]
        self.result_mar_method.append([case.alarm_start_time, case.rc, mar])
        
        # self.save_tmp_mar(case, mar)
        return mar

    def process_result_server(self, res_df, case, score_col = 'final_score'):
        adjust = - min(res_df[score_col]) if len(res_df[score_col]) else 0 
        res_df['final_score_adjusted'] = res_df[score_col] + adjust
        res_df.sort_values('final_score_adjusted', ascending = False).to_csv(os.path.join(self.result_dir, f'server-{case.alarm_start_time}.csv'), index = False)
        
        result_servers = {row.name: row.final_score_adjusted for row in res_df.itertuples()}
        result_servers = list(dict(sorted(result_servers.items(), key=lambda x: x[1], reverse=True)).keys())
        mar = [result_servers.index(i)+1 if i in result_servers else len(result_servers)+(case.server_num-len(result_servers))/2 for i in case.rc]
        self.result_mar.append([case.alarm_start_time, case.rc, mar])

        self.save_tmp_mar(case, mar)
        return mar

    def save_final_result(self, tp = 'server'):
        result, df = self.get_final_result(tp)

        save_path = f'result_mar_{tp}.csv'
        if os.path.exists(save_path):
            df_old = pd.read_csv(save_path)
            df.drop(columns='root_cause', inplace=True)
            df = df.merge(df_old, on = 'alarm_start_time', how = 'outer')
        df.to_csv(save_path, index=False)


    def get_final_result(self, tp = 'server'):
        res_df = self.result_mar if tp == 'server' else self.result_mar_method
        df = pd.DataFrame(res_df, columns=['alarm_start_time', 'root_cause', f'mar_{self.result_dir}'])
        topk = [1, 2, 3, 4, 5]
        avgtopk = [5]
        result = {}
        for k in topk:
            result_k = []
            rank = []
            for i, row in df.iterrows():
                if type(row[f'mar_{self.result_dir}']) == str: row[f'mar_{self.result_dir}'] = eval(row[f'mar_{self.result_dir}'])
                if type(row['root_cause']) == str: row['root_cause'] = eval(row['root_cause'])
                rank.extend([i for i in row[f'mar_{self.result_dir}'] if i>=0])
                numerator = len([i for i in row[f'mar_{self.result_dir}'] if (i>=1 and i<=k)])
                denominator = min(k,len(row['root_cause']))
                result_k.append(numerator/denominator)
            result[f'AC@{k}'] = round(np.sum(result_k)/df.shape[0],3)
            # result[k] = df[(df[f'mar_{self.result_dir}'] <= k+0.5) & (df[f'mar_{self.result_dir}'] > 0)].shape[0] / df.shape[0]
        for k in avgtopk:
            result[f'Avg@{k}'] = round(np.mean([result[f'AC@{i}'] for i in range(1,k+1)]),3)

        result['MAR'] = round(np.mean(rank),3)
        result['MRR'] = round(np.mean([1/i for i in rank]),3)
        print(result)
        return result, df
    
