import sys
sys.path.append('/home/wangs/rs/lib')
import ff
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from IPython.display import display


def get_overnight_turnover(code):
    result={}
    try:
        data=ff.read_min(code)
        # 首先转化为时间数据
        data.index=pd.to_datetime(data.index)
        # 提取第一分钟数据
        data=data.groupby(data.index.date).first()['volume']
        code_share=float_share.loc[code,:]
        # index转换
        data_date=data.index
        data.index=pd.to_datetime(data_date).strftime('%Y%m%d')
        # 开盘集合竞价换手率=第一分钟换手率/总换手率,除100
        result[code]=(data/code_share)/100
    except:
        result[code]=np.nan
        return result
    return result

def get_itaTurn20(code):
    result={}
    data=turnover.loc[code,:]
    result[code]=(data.rolling(window=20).mean()).shift(1)
    return result
    
def get_OvernightSmart(data):
    data['OvernightSmart']=(data["overnight_ret"]-data["overnight_ret"].min())/(data["overnight_ret"].max()-data["overnight_ret"].min())/data['overnight_turnover']  #隔夜聪明钱
    return data.sort_values('OvernightSmart')[:4].intraday_turnover_front.mean()
def get_CTR(code):
    result={}
    # 筛选数据
    data=pd.concat([intraday_turnover_front.loc[code,:],overnight_turnover.loc[code,:],overnight_ret.loc[code,:]],axis=1)
    data.columns=['intraday_turnover_front','overnight_turnover',"overnight_ret"]
    result[code]=(data['intraday_turnover_front'].rolling(window=20).apply(lambda x:get_OvernightSmart(data.loc[x.index]))).shift(1)
    return result

def get_JumpCTR_OvernightSmart(data):
    data['OvernightSmart']=(data["overnight_ret_next"]-data["overnight_ret_next"].min())/(data["overnight_ret_next"].max()-data["overnight_ret_next"].min())/data['overnight_turnover_next']  #隔夜聪明钱
    return data.sort_values('OvernightSmart')[:3].intraday_turnover.sum()
def get_JumpCTR(code):
    result={}
    # 筛选数据
    data=pd.concat([intraday_turnover.loc[code,:],overnight_turnover_next.loc[code,:],overnight_ret_next.loc[code,:]],axis=1)
    data.columns=['intraday_turnover','overnight_turnover_next',"overnight_ret_next"]
    result[code]=(data['intraday_turnover'].rolling(window=20).apply(lambda x:(get_JumpCTR_OvernightSmart(data.loc[x.index])+data.loc[x.index].iloc[-1,0])/4)).shift(1)
    return result
    
def main():
    from multiprocessing import Pool,Manager
    from tqdm import tqdm
    
    fname_1 = 'itaTurn20'
    float_share=ff.read('float_share')*ff.filter0
    codes=ff.cl
    with Pool(24) as p:
        res_lst_get_overnight_turnover = list(tqdm(p.imap(get_overnight_turnover, codes), total=len(codes)))
        p.close()
        p.join()
    # 开盘集合竞价换手率
    overnight_turnover=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_get_overnight_turnover}).T
    # 日内换手率=当日总换手率'turnover_rate'-开盘集合竞价换手率第一分钟换手率
    intraday_turnover=turnover-overnight_turnover
    # 对overnight_turnover大小进行变换
    overnight_turnover=turnover-intraday_turnover

    with Pool(24) as p:
        res_lst_itaTurn20 = list(tqdm(p.imap(get_itaTurn20, codes), total=len(codes)))
        p.close()
        p.join()
    mv=ff.read('total_mv')
    itaTurn20_un=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_itaTurn20}).T
    itaTurn20 = extend.spread_reg(itaTurn20_un, mv.loc[itaTurn20_un.index,itaTurn20_un.columns],ind=False) # ind=True为同时进行市值与行业中性化
    ff.save(fname_1,(itaTurn20*ff.filter0).dropna(how='all',axis=1))


    fname_2 = 'CTR'
    # 前一个交易日的日内换手率
    intraday_turnover_front=intraday_turnover.shift(1,axis=1) 
    with Pool(24) as p:
        res_lst_get_CTR = list(tqdm(p.imap(get_CTR, codes), total=len(codes)))
        p.close()
        p.join()
    CTR_un=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_get_CTR}).T
    CTR = extend.spread_reg(CTR_un, mv.loc[CTR_un.index,CTR_un.columns],ind=False) # ind=True为同时进行市值与行业中性化
    ff.save(fname_2,(CTR*ff.filter0).dropna(how='all',axis=1))


    fname_3 = 'JumpCTR'   
    overnight_turnover_next=overnight_turnover.shift(-1,axis=1)#次日隔夜换手率
    overnight_ret_next=overnight_ret.shift(-1,axis=1)#次日隔夜收益率
    with Pool(24) as p:
        res_lst_get_JumpCTR = list(tqdm(p.imap(get_JumpCTR, codes), total=len(codes)))
        p.close()
        p.join()
    JumpCTR = extend.spread_reg(JumpCTR_un, mv.loc[JumpCTR_un.index,JumpCTR_un.columns],ind=False) # ind=True为同时进行市值与行业中性化
    ff.save(fname_3,(JumpCTR*ff.filter0).dropna(how='all',axis=1))
    
if __name__ == '__main__':
    main()