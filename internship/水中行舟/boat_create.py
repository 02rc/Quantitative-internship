import sys
sys.path.append('/home/wangs/rs/lib')
'''sys.path.append('/home/wangs/rs/lwm/lib')'''
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib as mpl
mpl.rc("font", family='Droid Sans Fallback', weight="bold")
import extend

start, end = '20200102', '20240227'
start_time=pd.to_datetime(start)
end_time=pd.to_datetime('20240228')
read_data = lambda keyword:((ff.read(keyword) * ff.read('post') * ff.filter0).loc[:, start:end] 
                            if keyword in ('open', 'close', 'high', 'low')
                           else (ff.read(keyword) * ff.filter0).loc[:, start:end])
close,open,circ_mv,mv=read_data('close'),read_data('open'),read_data('circ_mv'),read_data("total_mv")
intraday_ret=close/open-1
codes=ff.cl
dates=list(pd.Series(index=ff.idt)[start:end].index)

def get_reasonable_ret(code):
    result={}
    data=intraday_ret.loc[code,:]
    result[code]=data.rolling(window=20,min_periods=1).mean()
    return result

with Pool(8) as p:
    res_lst_reasonable_ret = list(tqdm(p.imap(get_reasonable_ret, codes), total=len(codes)))
reasonable_ret=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_reasonable_ret}).T

reasonable_ret.columns=pd.to_datetime(reasonable_ret.columns)
def get_diff_vol(code):
    reasonable_ret_code=reasonable_ret.loc[code,:]
    try:
        data=ff.read_min(code)
        data.index=pd.to_datetime(data.index)
        data=data[start_time:end_time]#左闭右开
        if len(data)==0:
            pass
        else:
            open_data=data.groupby(data.index.date)['open'].first()
            data['open_ret']=(data['close']/open_data[data.index.date].values)-1
            data['h_l']=(data['open_ret']>reasonable_ret_code[data.index.date].values).replace({True:1,False:-1})
            mon_data=data.groupby(data.index.date).apply(lambda x:(x['h_l']*x['money']).sum())
            # index转换
            mon_data_date=mon_data.index
            mon_data.index=pd.to_datetime(mon_data_date).strftime('%Y%m%d')
            diff_vol=mon_data/circ_mv.loc[code,:]
            diff_vol.name=code
            return diff_vol
    except:
        pass

with Pool(8) as p:
    res_lst_diff_vol = list(tqdm(p.imap(get_diff_vol, codes), total=len(codes)))
diff_vol=pd.concat(res_lst_diff_vol,join='outer',axis=1).T

def get_with_flow(date):
    result={}
    try:
        data=diff_vol.loc[:, :date].iloc[:, -20:]
        spearman_corr=abs((data.T).corr(method='spearman'))
        np.fill_diagonal(spearman_corr.values, np.nan)
        result[date]=spearman_corr.mean()
        return result
    except:
        result[date]=np.nan
        return result

day_data_close = pd.read_pickle('/mydata2/wangs/data/fmins/close.pk')
day_data_close.index=pd.to_datetime(day_data_close.index)
day_data_close=day_data_close[start_time:end_time]
min_ret=day_data_close.groupby(day_data_close.index.date).apply(lambda x:(x-x.shift(1))/x.shift(1))
min_diff=min_ret.std(axis=1)
mean_min_diff=min_diff.groupby(min_diff.index.date).mean()
mean_min_diff.index=pd.to_datetime(mean_min_diff.index)
non_diff_min=min_diff[min_diff<mean_min_diff[min_diff.index.date].values].index

def get_date_money(code):
    try:
        data=ff.read_min(code)['money']
        data.index=pd.to_datetime(data.index)
        data=data[start_time:end_time]
        if len(data)!=0:
            data.name=code
            return data
        else:
            pass
    except:
        pass

with Pool(8) as p:
    res_lst_date_money = list(tqdm(p.imap(get_date_money, codes), total=len(codes)))
date_money=pd.concat(res_lst_date_money,join='outer',axis=1).T

def get_non_diff_min_new(date):
    date_time=pd.to_datetime(date)
    date_min=non_diff_min[non_diff_min.date==date_time]
    return date_min
with Pool(8) as p:
    res_lst_non_diff_min_new = list(tqdm(p.imap(get_non_diff_min_new, dates), total=len(dates)))
non_diff_min_new=pd.Series(res_lst_non_diff_min_new,index=dates)

# 减小计算量,数据预处理
def get_date_money_new(date):
    date_time=pd.to_datetime(date)
    data=(date_money.loc[:,date_money.columns.date==date_time]).dropna(how='all',axis=0)
    return data
with Pool(8) as p:
    res_lst_date_money_new = list(tqdm(p.imap(get_date_money_new, dates), total=len(dates)))
date_money_new=pd.Series(res_lst_date_money_new,index=dates)

def get_out_flock(date):
    result={}
    date_min=non_diff_min_new[date]
    pearson_corr=abs(((date_money_new[date].loc[:,date_min]).T).corr(method='pearson'))
    np.fill_diagonal(pearson_corr.values, np.nan)    
    result[date]=pearson_corr.mean()
    return result  

def get_out_flock(date):
    result={}
    date_min=non_diff_min_new[date]
    pearson_corr=abs(((date_money_new[date].loc[:,date_min]).T).corr(method='pearson'))
    np.fill_diagonal(pearson_corr.values, np.nan)    
    result[date]=pearson_corr.mean()
    return result   

with Pool(8) as p:
    res_lst_out_flock = list(tqdm(p.imap(get_out_flock, dates), total=len(dates)))
out_flock=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_out_flock})

def get_factor_out_flock(code):
    result={}
    # 有的code不在index中
    try:
        data=out_flock.loc[code,:]
        result[code]=(data.rolling(window=20,min_periods=20).mean()+data.rolling(window=20,min_periods=20).std())*0.5
        return result
    except:
        pass


def main():
    with Pool(8) as p:
        res_lst_with_flow = list(tqdm(p.imap(get_with_flow, dates), total=len(dates)))
    with_flow=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in res_lst_with_flow})
    with_flow.iloc[:,0:19]=np.nan

    with_flow_new = extend.spread_reg(with_flow,mv.loc[:,dates],ind=True) 
    ff.save('with_flow_RC',(with_flow_new*ff.filter0).dropna(how='all',axis=1))    

    with Pool(8) as p:
        res_lst_factor_out_flock = list(tqdm(p.imap(get_factor_out_flock, codes), total=len(codes)))
    cleaned_res_lst_factor_out_flock = [x for x in res_lst_factor_out_flock if x is not None]
    factor_out_flock=pd.DataFrame({list(i.keys())[0]: list(i.values())[0] for i in cleaned_res_lst_factor_out_flock}).T
    factor_out_flock.iloc[:,0:19]=np.nan
    factor_out_flock_new = extend.spread_reg(factor_out_flock,mv.loc[:,dates],ind=True) 
    ff.save('factor_out_flock_RC',(factor_out_flock_new*ff.filter0).dropna(how='all',axis=1))    

    boat_water=(factor_out_flock-with_flow)/2
    boat_water_new = extend.spread_reg(boat_water,mv.loc[:,dates],ind=True) 
    ff.save('boat_water_RC',(boat_water_new*ff.filter0).dropna(how='all',axis=1))    

if __name__ == '__main__':
    main()