# %load neverse_create.py
import sys
sys.path.append('/home/wangs/rs/lib')
import ff
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool
from tqdm import tqdm
import extend
import os

'''
名称: extreme_nervous_RC
来源: 20221213-方正证券-多因子选股系列研究之八：显著效应、极端收益扭曲决策权重和“草木皆兵”因子
作者: RC
构造方法:
1. 构建思路：个股择时信号 + 换手变化 = 特殊时刻的加速换手因子
2. 构建辅助因子：
i、惊恐度:  “偏离项”：计算个股收益率与中证全指收益率的差值，再取绝对值；“基准项”：计算个股收益率的绝对值，加市场收益率的绝对值，再加 0.1。“偏离项”除以“基准项”，得到该股票在该日的“惊恐度”。
ii、波动率：全天每分钟收益率求标准差
iii、个人投资者交易占比：将单笔成交金额小于 4 万元的交易，视为个人投资者交易。计算每天个股个人投资者卖出和买入的金额均值，再除以个股的当日总体成交金额，得到当日个股的个人投资者交易比。
iv、衰减后的“惊恐度”：将 t 日的惊恐度，减去 t-1 日和 t-2 日的“惊恐度”的均值，得到一个差值，将该差值为负的交易日的数据都替换为空值，记为衰减后的“惊恐度”。
3. 因子计算：
i、“原始惊恐”因子：每日的“惊恐度”与每日的收益率相乘作为加权决策分，计算过去 20 个交易日的“加权决策分”的均值和标准差，分别记为“惊恐收益”因子和“惊恐波动”因子，二者等权合成为“原始惊恐”因子。
ii、“波动率加剧-惊恐”因子：每日的“惊恐度”、波动率与收益率相乘作为加权决策分，计算过去 20 个交易日的“加权决策分”的均值和标准差，记为“波动率加剧-惊恐收益”因子和“波动率加剧-惊恐波动”因子，二者等权合成为“波动率加剧-惊恐”因子。
iii、“个人投资者交易比-惊恐”因子:每天的个人投资者交易比、“惊恐度”和收益率相乘作为加权决策分，计算过去 20 个交易日的“加权决策分”的均值和标准差，记为“个人投资者交易比-惊恐收益”因子和“个人投资者交易比-惊恐波动”因子，二者等权合成为“个人投资者交易比-惊恐”因子。
iv、“注意力衰减-惊恐”因子。每天的衰减后的“惊恐度”和收益率相乘作为加权决策分，计算过去 20 个交易日的“加权决策分”的均值和标准差，记为“注意力衰减-惊恐收益”因子和“注意力衰减-惊恐波动”因子，二者等权合成为“注意力衰减-惊恐”因子。
v、“草木皆兵”因子：将每天的衰减后的“惊恐度”、波动率、个人投资者者交易比、收益率相乘，作为当日的加权决策分，分别计算过去 20 日的加权决策分的均值和标准差，记为“草木皆兵-收益”因子和“草木皆兵-波动”因子，并将二者等权合成为“草木皆兵”因子。
'''
start, end = '20170104', '20230418'
def read_data(name,start_day=start,end_day=end):
    if name in ('open', 'close', 'high', 'low'):
        result = (ff.read(name) * ff.read('post') * ff.filter0).loc[ff.filter0.index, ff.filter0.loc[start_day:end_day].columns]
    else:
        result = (ff.read(name) * ff.filter0).loc[ff.filter0.index, ff.filter0.loc[start_day:end_day].columns]
    return result
    
def get_panic_degree(_array1,_array2):
    _deviation = abs(_array1-_array2[np.newaxis,:])
    _base = abs(_array1) + abs(_array2[np.newaxis,:]) + 0.1
    _panic_degree = _deviation/_base
    return _panic_degree

def get_decrease_panic_degree(_panic_degree): # n_stock,n_day
    num1, num2 = _panic_degree.shape
    _decrease_panic_degree = np.full((num1, num2), np.nan)
    _min_panic_degree = (_panic_degree[:,1:-1] + _panic_degree[:,:-2])*0.5
    _decrease_panic_degree[:,2:] = _panic_degree[:,2:] - _min_panic_degree
    _decrease_panic_degree = np.where(_decrease_panic_degree < 0,np.nan,_decrease_panic_degree)
    return _decrease_panic_degree # n_stock,n_day
    
def get_ret_vol(para):
    stock_name,date_lst = para
    try:
        stock_close_data = ff.read_min(stock_name).loc[date_lst[0]:date_lst[-1],'close']
        stock_close_data.index = pd.to_datetime(stock_close_data.index)
        stock_ret_data = stock_close_data.groupby(stock_close_data.index.date).pct_change()
        stock_volatility_data = stock_ret_data.groupby(stock_ret_data.index.date).std()
        stock_volatility_data.index = pd.to_datetime(stock_volatility_data.index)
        stock_volatility_data.index = stock_volatility_data.index.strftime('%Y%m%d')
        stock_volatility_data_match = stock_volatility_data.reindex(date_lst) 
        result = stock_volatility_data_match.values # n_date
    except:
        result = np.full(len(date_lst),np.nan) # n_date
    return result

def get_extreme_nervous(start,end,n = 20):
    # 数据读取：收益率（全天+分钟）、中证全指收益率
    
    # 得到收益率
    stock_ret_day = ff.rets_all.loc[:,start:end] # n_stock,n_day
    stock_lst = stock_ret_day.index
    date_lst = stock_ret_day.columns
    # 得到衰减-惊恐
    index_ret_day = pd.read_pickle('/mydata2/wangs/data/feature/zz1000.pk')['return']
    index_ret_day = index_ret_day.reindex(index = date_lst) # n_stock,n_day 
    panic_degree = get_panic_degree(stock_ret_day.values,index_ret_day.values) # n_stock,n_day 
    decrease_panic_degree = get_decrease_panic_degree(panic_degree) # n_stock,n_day
    # 得到波动率
    para_lst = [(stock_name,date_lst) for stock_name in stock_lst]
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(get_ret_vol,para_lst),total = len(stock_lst))) # n_stock,n_day 
    _volatility_array = np.array(res_lst) # n_stock,n_day
    # 得到个人持股占比
    hold_percent = (ff.read('buy_sm_amount') + ff.read('sell_sm_amount')) / (ff.read('amount')*2/10)
    hold_percent_match = hold_percent.reindex(columns = date_lst, index = stock_lst)

    # 计算因子
    weighted_decision_score = decrease_panic_degree * _volatility_array * hold_percent_match.values * stock_ret_day.values # n_stock,n_day 
    weighted_decision_score_rolling = ff.rolling_window(weighted_decision_score,n)# n_stock,n_day-n + 1,n
    weighted_decision_score_mean = np.nanmean(weighted_decision_score_rolling,axis = -1)# n_stock,n_day-n + 1
    weighted_decision_score_std = np.nanstd(weighted_decision_score_rolling,axis = -1)# n_stock,n_day-n + 1
    extreme_nervous_array = (weighted_decision_score_mean+weighted_decision_score_std)*0.5# n_stock,n_day-n + 1
    extreme_nervous = pd.DataFrame(extreme_nervous_array,columns = date_lst[n-1:],index = stock_lst)

    return extreme_nervous

def check(new_f, org_f):
    new_f = new_f.round(6)
    org_f = org_f.round(6)
    syms_set = sorted(list((set(new_f.index) & set(org_f.index))))
    dates_set = sorted(list((set(new_f.columns) & set(org_f.columns))))
    ident = new_f.loc[syms_set,dates_set].equals(org_f.loc[syms_set,dates_set])
    if ident:
        return True
    else:
        print(new_f.loc[syms_set,dates_set].compare(org_f.loc[syms_set,dates_set]))
        return False


def version_reserve(new_f,f_name):
    if os.path.exists(f_name + '_version_reserver'):
        pass
    else:
        os.makedirs(f_name + '_version_reserver')
    
    folder_path = f_name + '_version_reserver'

    # 获取文件夹中所有文件的路径和修改时间
    files = [(join(folder_path, file), getmtime(join(folder_path, file)))
             for file in os.listdir(folder_path)
             if os.path.isfile(join(folder_path, file))]

    # 检查文件数量，如果超过 5 个，则删除最旧的文件
    if len(files) > 5:
        os.remove(files[0][0])  # 删除最旧的文件
        print(f"已删除文件：{files[0][0]}")
    td = str(datetime.datetime.today())[:10].replace('-','')
    
    new_f.to_hdf(f'{f_name}_version_reserver/{f_name}_{td}' + '.h5', key='data')
    print(f"已保存文件：{f_name}_{td}")
    
def main():
    n = 20
    date_lst = ff.read('close').columns
    end = date_lst[-1]
    update_win = 20
    start = date_lst[-(n+update_win+10)] 
    
    extreme_nervous_RC = get_extreme_nervous(start,end,n)
    mv = ff.read('total_mv')
    mv_match = mv.reindex(index = extreme_nervous_RC.index,columns = extreme_nervous_RC.columns)
    extreme_nervous_RC_neu = extend.spread_reg(extreme_nervous_RC,mv_match, ind=True) # ind=True为同时进行市值与行业中性化
    ff.save('extreme_nervous_RC',extreme_nervous_RC_neu.shift(1,axis=1)*ff.filter0)

    # update
    his_extreme_nervous_RC = ff.read('extreme_nervous_RC').to_dict()
    new_extreme_nervous_RC = (extreme_nervous_RC_neu.shift(1,axis = 1) * ff.filter0).iloc[:,-update_win:].to_dict()
    his_extreme_nervous_RC.update(new_extreme_nervous_RC)
    his_extreme_nervous_RC = pd.DataFrame(his_extreme_nervous_RC)

    # check
    assert check(ff.read('extreme_nervous_RC'),his_extreme_nervous_RC)
    version_reserve(his_extreme_nervous_RC,'extreme_nervous_RC') # 旧版本保留
    ff.save('extreme_nervous_RC',his_extreme_nervous_RC)

    
if __name__ == '__main__':
    main()