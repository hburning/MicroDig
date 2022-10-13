import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import time
import glob
import gc
from utils import ts2date, date2ts, date2datetime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 默认的时间格式
time_begin = datetime.datetime(2022,5,27)

files = glob.glob('case2/*.csv')
df_res = pd.DataFrame()

for fi in tqdm(files):
    gc.collect()
    df = pd.read_csv(fi).fillna('')
    df.drop_duplicates(inplace=True)
    
    #adjust the caller and callee
    df = df[df['operationName'].apply(lambda x: False if re.search(r'health', x, flags = re.I) else True)]
    df.loc[(df['processServiceName'] == 'logservice') & (df['operationName'] == 'HTTP POST'), 'peerName'] = 'es'
    df['callee_service'] = df.apply(lambda x: x['service'] if x['service']!='' else x['peerName'], axis = 1)

    df.loc[:,'callee'] = df.apply(lambda x: x['callee_service'] + '|' + x['method'] , axis = 1)
    df.loc[:,'startTime_min'] = df.apply(lambda x: (date2ts(x['startTime']) // 60) * 60, axis = 1)
    df.loc[:, 'caller'] = ''
    df.loc[df['processServiceName'] == 'frontend', 'caller'] = 'frontend|'
    df.loc[(df['processServiceName'] == 'frontend') & (df['peerIP'] == '127.0.0.6'), 'caller'] = 'frontend_exter|'
    df.loc[(df['processServiceName'] == 'frontend') & (df['peerIP'] == '127.0.0.6'), 'callee'] = 'frontend|'

    span_index = dict(zip(df['spanId'].values, df.index))

    #find the parent
    for row in df.itertuples():
        if row.processServiceName == 'frontend':
            continue
        if row.spanKind == 'client':
            parent_server_id = span_index.get(row.parentSpanId)
            if parent_server_id:
                parent_client_id = span_index.get(df.loc[parent_server_id]['parentSpanId'])
                if parent_client_id and re.search(row.processServiceName, df.loc[parent_client_id]['callee'], flags = re.I):
                    df.loc[row[0], 'caller'] = df.loc[parent_client_id]['callee']

    df = df[(df['spanKind'] == 'client') | ((df['processServiceName'] == 'frontend') & (df['peerIP'] == '127.0.0.6'))]
    # print('df.shape:', df.shape, end = '')
    # df = df[df['callee_service'] != '']
    # print('df.shape filtered:', df.shape)
    df.loc[df['processServiceName'].str.contains('checkoutservice') & (df['caller'] == ''), 'caller'] = 'hipstershop.CheckoutService|PlaceOrder'
    print('have no parent:', df[((df['caller'] == '') | (df['callee'] == ''))]['processServiceName'].value_counts())
    df = df[(df['caller'] != '') & (df['callee'] != '')]
    # print('df.shape filtered:', df.shape)

    groups = df.groupby(['caller', 'callee', 'startTime_min'])
    result = []
    for i, g in groups:
        error_num = g[g['error'] == True].shape[0]
        error_rate = g[g['error'] == True].shape[0]/g.shape[0]
        result.append([i[0], i[1], int((ts2date(i[2]) - time_begin).days*1440+(ts2date(i[2]) - time_begin).seconds/60), g['duration'].sum()/g.shape[0], error_num, error_rate, g.shape[0]])
    df_new = pd.DataFrame(result, columns=['caller', 'callee', 'time_min', 'duration', 'error_min', 'error_rate', 'request_min'])
    df_res = pd.concat([df_res, df_new])
    print("df_new:", df_res.shape)
    
outf = 'output.csv'
if os.path.exists(outf):
    df_old = pd.read_csv(outf)
    df_res = pd.concat([df_res, df_old])

df_res.to_csv(outf, index = False)




