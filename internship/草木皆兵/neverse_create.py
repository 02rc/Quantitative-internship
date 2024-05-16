# %load neverse_create.py
import sys
sys.path.append('/home/wangs/rs/lib')
import ff
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from IPython.display import display
import tushare as ts
from multiprocessing import Pool
from tqdm import tqdm
import extend
pro = ts.pro_api('5b6a2c5b17b9572fe089a0802765f8b6434e72c51572718b1d37c302')

start, end = '20200101', '20240201'
zz_total = pro.index_daily(ts_code='000985.SH', start_date='2020-01-01')    
zz_total_ret=zz_total['close'].pct_change(-1)
zz_total_ret.index=zz_total['trade_date']
zz_total_ret=zz_total_ret.loc[zz_total_ret.index.isin(ff.idt)]
codes=ff.cl


def read_data(keyword):
    if keyword in ('open', 'close', 'high', 'low'):
        return (ff.read(keyword) * ff.read('post') * ff.filter0).loc[:, start:end] 
    else:
        return (ff.read(keyword) * ff.filter0).loc[:, start:end]
        
close=read_data('close')
ret=(close-close.shift(1,axis=1))/close.shift(1,axis=1)

# 惊恐度
def get_panic_degree(code):
    result={}
    result[code]= abs(ret.loc[code,:]-zz_total_ret)/(abs(ret.loc[code,:])+abs(zz_total_ret)+0.1)
    return result

with Pool(24) as p:
    res_lst_panic_degree = list(tqdm(p.imap(get_panic_degree, codes), total=len(codes)))
panic_degree=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_panic_degree}).T  

def get_primal_panic(code):
    result_1={}
    result_2={}
    result_3={}
    data=panic_degree.loc[code,:]*ret.loc[code,:]
    factor_ret=(data.rolling(window=20, min_periods=20).mean())
    factor_vol=(data.rolling(window=20, min_periods=20).std())
    result_1[code]=factor_ret
    result_2[code]=factor_vol
    result_3[code]=(factor_ret+factor_vol)*0.5
    return result_1,result_2,result_3


def get_volatility(code):
    result={}
    try:
        data=ff.read_min(code)['close']
        data.index=pd.to_datetime(data.index)
        data=data.groupby(data.index.date).apply(lambda x:((x-x.shift(1))/x.shift(1)).std())
        # index转换
        data_date=data.index
        data.index=pd.to_datetime(data_date).strftime('%Y%m%d')
        result[code]=data
    except:
        result[code]=np.nan
    return result
with Pool(24) as p:
    res_lst_volatility = list(tqdm(p.imap(get_volatility, codes), total=len(codes)))
volatility=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_volatility}).T
individual_trading_ratio=10*(read_data('buy_sm_amount')+read_data('sell_sm_amount'))/(read_data('amount')*2)
attenuated_panic_degree=panic_degree-(panic_degree.shift(1,axis=1)+panic_degree.shift(2,axis=1))/2
attenuated_panic_degree[attenuated_panic_degree< 0]=np.nan

def get_extreme_nervous(code):
    result_1={}
    result_2={}
    result_3={}
    data=ret.loc[code,:]*attenuated_panic_degree.loc[code,:]*volatility.loc[code,:]*individual_trading_ratio.loc[code,:]
    extreme_nervous_ret=(data.rolling(window=20, min_periods=5).mean())
    extreme_nervous_vol=(data.rolling(window=20, min_periods=5).std())
    result_1[code]=extreme_nervous_ret
    result_2[code]=extreme_nervous_vol
    result_3[code]=(extreme_nervous_ret+extreme_nervous_vol)*0.5
    return result_1,result_2,result_3    

def get_extreme_nervous_week(code):
    result_1={}
    result_2={}
    result_3={}
    data=ret.loc[code,:]*attenuated_panic_degree.loc[code,:]*volatility.loc[code,:]*individual_trading_ratio.loc[code,:]
    extreme_nervous_ret_week=(data.rolling(window=5, min_periods=4).mean())
    extreme_nervous_vol_week=(data.rolling(window=5, min_periods=4).std())
    result_1[code]=extreme_nervous_ret_week
    result_2[code]=extreme_nervous_vol_week
    result_3[code]=(extreme_nervous_ret_week+extreme_nervous_vol_week)*0.5
    return result_1,result_2,result_3    


def main():
    from multiprocessing import Pool,Manager
    from tqdm import tqdm

    mv=read_data("total_mv")
  
    with Pool(24) as p:
        res_lst_get_primal_panic = list(tqdm(p.imap(get_primal_panic, codes), total=len(codes)))
    res_lst_factor_ret,res_lst_factor_vol, res_lst_primal_panic = zip(*res_lst_get_primal_panic)
    primal_panic=((pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_primal_panic}).T)*ff.filter0).dropna(how='all',axis=1)
    primal_panic_new = extend.spread_reg(primal_panic.loc[:,start:end],mv.loc[:,start:end],ind=True) 
    ff.save('primal_panic_RC',primal_panic_new.shift(1,axis=1))

    with Pool(24) as p:
        res_lst_get_extreme_nervous = list(tqdm(p.imap(get_extreme_nervous, codes), total=len(codes)))
    res_lst_extreme_nervous_ret,res_lst_extreme_nervous_vol, res_lst_extreme_nervous = zip(*res_lst_get_extreme_nervous)
    extreme_nervous=((pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_extreme_nervous}).T)*ff.filter0).dropna(how='all',axis=1)
    extreme_nervous_new = extend.spread_reg(extreme_nervous.loc[:,start:end],mv.loc[:,start:end],ind=True)   
    ff.save('extreme_nervous_RC',extreme_nervous_new.shift(1,axis=1))

    
    with Pool(24) as p:
        res_lst_get_extreme_nervous_week = list(tqdm(p.imap(get_extreme_nervous_week, codes), total=len(codes)))
    res_lst_extreme_nervous_ret_week,res_lst_extreme_nervous_vol_week, res_lst_extreme_nervous_week = zip(*res_lst_get_extreme_nervous_week)
    extreme_nervous_week=((pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_extreme_nervous_week}).T)*ff.filter0).dropna(how='all',axis=1)
    extreme_nervous_week_new = extend.spread_reg(extreme_nervous_week.loc[:,start:end], mv.loc[:,start:end],ind=True) 
    ff.save('extreme_nervous_week_RC',extreme_nervous_week_new.shift(1,axis=1))
    
if __name__ == '__main__':
    main()