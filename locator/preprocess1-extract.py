import pickle
import snappy
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json
import base64
import glob

def base64_to_hex(payload_base64):
    bytes_out = base64.b64decode(payload_base64)    
    str_out = bytes_out.hex()                       
    return str_out

def convert_to_df(trace_list):
    df = pd.DataFrame()

    traceId = []
    spanId, parentSpanId = [], []
    serviceName, operationName = [], []
    startTime, duration, nanosecond = [], [], []
    status  = []
    spanKind, peerName, httpStatus, cur_errorState, errorState, error, peerIP = [], [], [], [], [], [], []
    method, service = [], []

    for span in tqdm(trace_list):
            traceId.append(base64_to_hex(span['traceId']))

            spanId.append(base64_to_hex(span['spanId']))
            if span.get('references'):
                parentSpanId.append(base64_to_hex(span['references'][0]['spanId']))
            else:
                parentSpanId.append(0)

            serviceName.append(span['process']['serviceName'])
            operationName.append(span['operationName'])


            # start_datetime = datetime.fromtimestamp(span['startTime'].seconds).strftime('%Y-%m-%d %H:%M:%S')
            try:
                time_obj = datetime.strptime(span['startTime'], '%Y-%m-%dT%H:%M:%S.%fZ')
            except Exception as e:
                time_obj = datetime.strptime(span['startTime'], '%Y-%m-%dT%H:%M:%SZ')
                
            start_datetime = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            startTime.append(start_datetime)
            # duration.append(span['duration']['nanos'] // 1000) # nanosecond to microsecond
            duration.append(float(span['duration'][:-1]) * 1e6)  # second to microsecond 
            # nanosecond.append(span['startTime']['nanos'])
            nanosecond.append(time_obj.microsecond * 1000)     # microsecond to nanosecond

            cur_status, cur_httpcode = -1, -1
            cur_kind, cur_peername, cur_errorState, cur_peerip = '', '', '', ''
            cur_error, cur_method, cur_service = 'false', '', ''
            for tag in span['tags']:
                key = tag['key']
                if key == 'rpc.grpc.status_code':
                    cur_status = int(tag.get('vInt64', -1))
                if key == 'span.kind':
                    cur_kind = tag['vStr']
                if key == 'net.peer.name':
                    cur_peername = tag['vStr']
                if key == 'net.peer.ip':
                    cur_peerip = tag['vStr']
                if key == 'http.status_code':
                    cur_httpcode = int(tag['vInt64'])
                if key == 'otel.status_code':
                    cur_errorState = tag['vStr']
                if key == 'error':
                    cur_error = tag['vBool']
                if key == 'rpc.service':
                    cur_service = tag['vStr']
                if key == 'rpc.method':
                    cur_method = tag['vStr']

            status.append(cur_status)
            spanKind.append(cur_kind)
            peerName.append(cur_peername)
            peerIP.append(cur_peerip)
            httpStatus.append(cur_httpcode)
            errorState.append(cur_errorState)
            error.append(cur_error)
            method.append(cur_method)
            service.append(cur_service)

    df['traceId'] = traceId
    df['spanId'] = spanId
    df['spanKind'] = spanKind
    df['peerName'] = peerName
    df['peerIP'] = peerIP
    df['parentSpanId'] = parentSpanId
    df['processServiceName'] = serviceName
    df['operationName'] = operationName
    df['service'] = service
    df['method'] = method
    df['startTime'] = startTime
    df['duration'] = duration
    df['nanosecond'] = nanosecond
    df['status'] = status
    df['httpStatus'] = httpStatus
    df['errorState'] = errorState
    df['error'] = error

    return df


fdir = './case2'
files = glob.glob(os.path.join(fdir, '*.pkl'))
for fi in files:
    print(fi)
    outf = os.path.join(fdir, f'{fi.split("/")[-1][:-4]}.csv')
    if os.path.exists(outf): continue
    trace = pickle.loads(snappy.decompress(open(fi,'rb').read()))
    trace_list = [json.loads(data) for data in tqdm(trace)]
    df = convert_to_df(trace_list)
    df.to_csv(outf, index = False)