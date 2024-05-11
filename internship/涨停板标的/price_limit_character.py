import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib as mpl
mpl.rc("font", family='Droid Sans Fallback', weight="bold")
from datetime import datetime
import os

# 涨停板为10%，剔除涨停板为5%，20%，30%的标的
# 做法：选出当日收益率小于或等于11%的股票（允许浮动）
event_ret = ((ff.rets_all<=0.11) & (ff.rets_all>=0.09)).replace({True:1,False:np.nan})
# 收盘涨停：
event_close = ((ff.read('close')-ff.read('up_limit'))==0).replace({True:1,False:np.nan})

up_data = event_ret*ff.read('up')*event_close

codes = up_data.index.tolist()
index_lst = up_data.columns
# 是否为一字涨停
# 一字涨停表示在涨停状态下，该股票的价格没有发生过任何交易，一直保持在涨停价的水平上，直到当日的交易结束
def get_always_up(code):
    result = pd.Series(np.nan,index = index_lst,name = code)
    dates_1 = up_data.loc[code,:].dropna().index
    if len(dates_1) == 0:
        pass
    else:
        datas = ff.read_min(code).loc[:,['close','high_limit']]
        for date1 in dates_1:
            try:
                date2 = str(int(date1)+1)
                date1_re = date1[:4] + '-' + date1[4:6] + '-' + date1[6:]
                date2_re = date2[:4] + '-' + date2[4:6] + '-' + date2[6:]
                daily_data = datas.loc[date1_re:date2_re] 
                op_values = (daily_data['close']-daily_data['high_limit']).values
                i = False
                j = True
                for value in op_values:
                    if value == 0:
                       i = True
                    elif i:
                        j = False
                if i and j:
                    result[date1] = 1
            except IndexError as e:
                pass
    return result

# 股票炸板即股票打开涨停板，或者打开跌停板
def get_break(code):
    result = pd.Series(np.nan,index = index_lst,name = code)
    dates_1 = up_data.loc[code,:].dropna().index
    if len(dates_1) == 0:
        pass
    else:
        datas = ff.read_min(code).loc[:,['close','high_limit']]
        for date1 in dates_1:
            try:
                date2 = str(int(date1)+1)
                date1_re = date1[:4] + '-' + date1[4:6] + '-' + date1[6:]
                date2_re = date2[:4] + '-' + date2[4:6] + '-' + date2[6:]
                daily_data = datas.loc[date1_re:date2_re] 
                op_values = (daily_data['close']-daily_data['high_limit']).values
                i = False
                j = False
                for value in op_values:
                    if value == 0:
                       i = True
                    elif i:
                        j = True
                if i and j:
                    result[date1] = 1
            except IndexError as e:
                pass
    return result

def get_time(code):
    result = pd.Series(name = code)
    dates_1 = up_data.loc[code,:].dropna().index
    if len(dates_1) == 0:
        pass
    else:
        datas = ff.read_min(code).loc[:,['close','high_limit']]
        for date1 in dates_1:
            try:
                date2 = str(int(date1)+1)
                date1_re = date1[:4] + '-' + date1[4:6] + '-' + date1[6:]
                date2_re = date2[:4] + '-' + date2[4:6] + '-' + date2[6:]
                daily_data = datas.loc[date1_re:date2_re] 
                up_time = daily_data[(daily_data['close']-daily_data['high_limit'])==0].index
                for _time in up_time:
                    result.loc[_time] = 1
            except IndexError as e:
                pass
    return result

def get_first_volume(code):
    result = pd.Series(np.nan,index = index_lst,name = code)
    dates_1 = up_data.loc[code,:].dropna().index
    if len(dates_1) == 0:
        pass
    else:
        datas = ff.read_min(code).loc[:,['close','high_limit','volume']]
        for date1 in dates_1:
            try:
                date2 = str(int(date1)+1)
                date1_re = date1[:4] + '-' + date1[4:6] + '-' + date1[6:]
                date2_re = date2[:4] + '-' + date2[4:6] + '-' + date2[6:]
                daily_data = datas.loc[date1_re:date2_re] 
                op_values = (daily_data['close']-daily_data['high_limit']).values
                volume_values = daily_data['volume'].values
                for i in range(len(op_values)):
                    if op_values[i] == 0:
                        result[date1] = volume_values[i]
                        break
            except IndexError as e:
                pass
    return result

def get_up_volume(code):
    result = pd.Series(np.nan,index = index_lst,name = code)
    dates_1 = up_data.loc[code,:].dropna().index
    if len(dates_1) == 0:
        pass
    else:
        datas = ff.read_min(code).loc[:,['close','high_limit','volume']]
        for date1 in dates_1:
            try:
                date2 = str(int(date1)+1)
                date1_re = date1[:4] + '-' + date1[4:6] + '-' + date1[6:]
                date2_re = date2[:4] + '-' + date2[4:6] + '-' + date2[6:]
                daily_data = datas.loc[date1_re:date2_re] 
                op_values = (daily_data['close']-daily_data['high_limit']).values
                volume_values = daily_data['volume'].values
                volume_total = 0
                for i in range(len(op_values)):
                    if op_values[i] == 0:
                        volume_total += volume_values[i]
                result[date1] = volume_total
            except IndexError as e:
                pass
    return result
def main():
    with Pool(36) as p:
        res_lst = list(tqdm(p.imap(get_always_up,codes),total=len(codes)))
    always_up_event = pd.concat(res_lst,axis = 1)
    with Pool(36) as p:
        res_lst = list(tqdm(p.imap(get_break,codes),total=len(codes)))
    break_event = pd.concat(res_lst,axis = 1)
    with Pool(36) as p:
        res_lst = list(tqdm(p.imap(get_time,codes),total=len(codes)))
    up_time = pd.concat(res_lst,axis = 1).sort_index()
    with Pool(36) as p:
        res_lst = list(tqdm(p.imap(get_first_volume,codes),total=len(codes)))
    first_volume = pd.concat(res_lst,axis = 1).sort_index()
    with Pool(36) as p:
        res_lst = list(tqdm(p.imap(get_up_volume,codes),total=len(codes)))
    up_volume = pd.concat(res_lst,axis = 1).sort_index()
    
if __name__ == '__main__':
    main()