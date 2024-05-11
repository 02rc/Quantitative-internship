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
import matplotlib.pyplot as plt

'''
择时策略——双均线策略：        
1. 短均线上穿长均线 (金叉)，且当前无持仓：买入；
2. 短均线下穿长均线 (死叉)，且当前持仓，卖出；
3. 其他情况，保持之前仓位;
4. 可以考虑控制回撤，单次亏损超过一定幅度平仓。
'''
def double_line_hold(para,sigma1 = -0.05,sigma2 = -0.05):
    para_re = np.reshape(para,(3,int(len(para)/3)))
    short_line,long_line,_close = para_re[0],para_re[1],para_re[2]
    len_line = len(short_line)
    hold_lst = [np.nan]*len_line 
    m = 0 # 辅助指标
    k = 0 # 长短均线上一时刻的位置关系(-1:long>short,1:long<short)
    for i in range(1, len_line):
        if m != 0:
            hold_lst[i] = m # 延续持仓状态
            if (m*(_close[i]/_close[index1]-1))<sigma1:#止损
                m = 0
            else:
                if (m*_close[i]) > (m*max_close):
                    max_close = _close[i]
                elif (m*(_close[i]/max_close-1)) < sigma2:#动态止盈
                    m = 0
        if long_line[i]>short_line[i]:
            if k ==1 :
                m = 1
                max_close = _close[i]  
                index1 = i
            k = -1
        elif long_line[i]<=short_line[i]:
            if k ==-1 :
                m = -1
                max_close = _close[i]  
                index1 = i
            k = 1            
    return hold_lst

def data_FM(para):
    _name,_fre,_start,_end = para
    _name_close = ff.read_binance(_name).c
    _name_close.index = pd.to_datetime(_name_close.index)
    if _start in _name_close.index:
        pass
    else:
        _name_close.loc[_start] = np.nan
    if _end in _name_close.index:
        pass
    else:
        _name_close.loc[_end] = np.nan
    _name_close_new = _name_close.resample(f'{_fre}T').ffill()
    para_close = _name_close_new.loc[_start:_end]
    para_ret = (_name_close_new/_name_close_new.shift(1) - 1).loc[_start:_end]
    return para_close.values ,para_ret.values

class double_line_strategy(object):
    def __init__(self,factor,N_1 = 1,N_2 = 20,fees = 0):
        if not isinstance(factor.index, pd.DatetimeIndex):
            factor.index = pd.to_datetime(factor.index)
        self.short_line = (factor.rolling(window = N_1,min_periods = 1,axis = 1).mean()).T.values
        self.long_line = (factor.rolling(window = N_2,min_periods = 1,axis = 1).mean()).T.values
        self.fre = (factor.index[1]-factor.index[0]).total_seconds() / 60
        self.start = factor.index[0]
        self.end = factor.index[-1]
        self.fees = fees
        self.names = factor.columns
        self.index = factor.index
        self.initial_para()
 
    def initial_para(self):
        _para_lst = [(_name,self.fre,self.start,self.end) for _name in self.names]
        with Pool(24) as p:
            ret_close_lst = list(p.imap(data_FM, _para_lst))    
        close_lst,ret_lst = zip(*ret_close_lst)
        self.ret = np.array(ret_lst)
        self.close = np.array(close_lst)

    def apply_strategy(self):
        _para_lst = np.concatenate((self.short_line,self.long_line,self.close), axis=1)
        with Pool(24) as p:
            _hold_lst = list(p.imap(double_line_hold, _para_lst))  
        _hold_arr = np.array(_hold_lst)
        _ret_lst = _hold_arr*self.ret
        _ret_mean_lst = np.nanmean(_ret_lst,axis=0)
        _ret_df = pd.Series(_ret_mean_lst,index = self.index)
        return _ret_df

    def performance(self):
        event_ret = self.apply_strategy().dropna()
        _all_ret_mean_lst = np.nanmean(self.ret,axis=0)
        _all_ret_df = pd.Series(_all_ret_mean_lst,index = self.index).loc[event_ret.index[0]:event_ret.index[-1]]
        over_ret = (event_ret - _all_ret_df).dropna()
        # 设置图形的尺寸和分辨率
        plt.figure(figsize=(10, 6), dpi=80)
        # 绘制折线图
        event_ret.cumsum().plot()
        _all_ret_df.cumsum().plot()
        over_ret.cumsum().plot()
        # 添加标题和标签
        plt.title('累积收益率')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        # 添加网格线
        plt.grid(True)
        # 添加图例
        plt.legend(['策略收益','平均收益','超额收益'])
        # 自动调整日期格式
        plt.gcf().autofmt_xdate()
        # 显示图形
        plt.show()

        print(ff.cal_returns((event_ret.groupby(event_ret.index.date).apply(lambda x:(x+1).prod() - 1)).fillna(0)))
        
    def op_factor(self):
        event_ret = self.apply_strategy().dropna()
        return ff.cal_returns((event_ret.groupby(event_ret.index.date).apply(lambda x:(x+1).prod() - 1)).fillna(0))
            
        