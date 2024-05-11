import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import extend
from datetime import datetime
import statsmodels.api as sm
import os
'''
名称: TPS_RC,SPS_RC
来源: 20230202-金工定期报告：TPS与SPS选股因子绩效月报-东吴证券
作者: RC
构造方法:
1. 构建辅助因子：i、日内收益率，日内收益率的构建方法为，今日收盘价减去今日开盘价的差除以昨日收盘价；ii、振荡幅度，振幅的构建方法为，今日最高价减去今日最低价的差除以昨日收盘价；iii:价格因子，影线差。构建方法为上影线减去下影线后除以昨日收盘价。
2. 用截面数据，换手率作为因变量，影线差 PLUS 作为自变量，使用一元线性回归的方法取得残差项。为Turn_dePLUS，20天移动平均得到：Turn_dePLUS20；
3. Turn_dePLUS 因子值回滚20天取标准差为STR_dePLUS
4. 用截面数据，PLUS 因子作为因变量，换手率作为自变量，进行横截面的一元线性回归，取得残差项，为 PLUS_deTurn。20天移动平均得到：PLUS_deTurn20；
5. TPS 全名为 Turn20 conformed by PLUS，构建方法是把 Turn20_dePLUS 与PLUS_deTurn20 相乘；SPS 全名为 STR conformed by PLUS，构建方法是把 STR_dePLUS 与 PLUS_deTurn20相乘
'''
def cleaned(date_Turn,date_PLUS):
    date_Turn_scale = (date_Turn-np.nanmean(date_Turn))/np.nanstd(date_Turn)
    date_PLUS_scale = (date_PLUS-np.nanmean(date_PLUS))/np.nanstd(date_PLUS)
    valid_indices = np.logical_and(~np.isnan(date_Turn_scale), ~np.isnan(date_PLUS_scale))
    date_Turn_valid = date_Turn[valid_indices]
    date_PLUS_valid = date_PLUS[valid_indices]
    PLUS_deTurn = np.full_like(date_Turn, np.nan)
    Turn_dePLUS = np.full_like(date_Turn, np.nan)
    # 防止所有位置均为空值的情况
    if not np.all(~valid_indices):
        date_Turn_valid_c = sm.add_constant(date_Turn_valid)
        date_PLUS_valid_c = sm.add_constant(date_PLUS_valid)
        
        PLUS_deTurn_model = sm.OLS(date_PLUS_valid, date_Turn_valid_c)
        Turn_dePLUS_model = sm.OLS(date_Turn_valid, date_PLUS_valid_c)
        
        PLUS_deTurn_results = PLUS_deTurn_model.fit()
        Turn_dePLUS_results = Turn_dePLUS_model.fit()    
    
        PLUS_deTurn[valid_indices] = PLUS_deTurn_results.resid  
        Turn_dePLUS[valid_indices] = Turn_dePLUS_results.resid  
    else:
        pass    
    return PLUS_deTurn,Turn_dePLUS

class TPS_SPS(object):
    def __init__(self,name,fre,n = 20):
        # n 为计算Turn时的回望参数
        self.n = n
        self.name = name
        # fre为最终计算出的因子频率
        _h_l_c_v = ff.read_binance(name)[['h','l','c','v']]
        _h_l_c_v.index = pd.to_datetime(_h_l_c_v.index)
        _h_l_c_v_re = _h_l_c_v.resample('1T').asfreq()
        # 为了便于数据处理，只考虑完整的日期
        _start_date = _h_l_c_v_re.index[0].date()  
        _end_date = _h_l_c_v_re.index[-1].date()  
        _keep_start_date = _start_date + pd.Timedelta(days=1)  
        _cleaned_h_l_c_v = _h_l_c_v_re.loc[_keep_start_date:_end_date].iloc[:-1]
        
        _v_values = _cleaned_h_l_c_v.v.resample(f'{fre}T').sum().values
        _h_values = _cleaned_h_l_c_v.h.resample(f'{fre}T').max().values
        _l_values = _cleaned_h_l_c_v.l.resample(f'{fre}T').min().values
        _c_values = _cleaned_h_l_c_v.c.resample(f'{fre}T').last().values

        self.index_lst = _cleaned_h_l_c_v.index.values[::fre]
        self.Turn = self.get_Turn(_v_values)
        self.PLUS = self.get_PLUS(_c_values,_h_values,_l_values)
        
    def get_Turn(self,_data):
        _data_slice = ff.rolling_window(_data,self.n)
        _data_day_mean = np.nanmean(_data_slice,axis = 1)
        _Turn = np.concatenate((np.full(self.n-1, np.nan),_data[self.n-1:]/_data_day_mean))
        return _Turn

    def get_PLUS(self,_data_c,_data_h,_data_l):
        _PLUS = (2*_data_c-_data_h-_data_l)/np.concatenate((np.full(1, np.nan),_data_c[1:]))
        return _PLUS

# 无法进行截面纯净化，考虑时间纯净化
    def rolling_result(self,_Turn,_PLUS,n2):
        try:
            _Turn_slice = ff.rolling_window(_Turn,n2)
            _Turn20 = np.nanmean(_Turn_slice,axis = 1)
            _STR = np.nanstd(_Turn_slice,axis = 1)
            _PLUS_slice = ff.rolling_window(_PLUS,n2)
            _PLUS20 = np.nanmean(_PLUS_slice,axis = 1)
            _TPS = _Turn20*_PLUS20
            _SPS = _STR*_PLUS20
            _TPS_se = pd.Series(_TPS,index = self.index_lst[n2-1:],name = self.name)
            _SPS_se = pd.Series(_SPS,index = self.index_lst[n2-1:],name = self.name)
        except ValueError as e:
            _TPS_se = pd.Series(index = self.index_lst[n2-1:],name = self.name)
            _SPS_se = pd.Series(index = self.index_lst[n2-1:],name = self.name)
        return _TPS_se,_SPS_se
        
    def factor_initial(self,n2 = 20):
        # n2 表示对20天做移动平均
        _TPS_se,_SPS_se = self.rolling_result(self.Turn,self.PLUS,n2)
        return _TPS_se,_SPS_se

    def factor_time_cleaned(self,n1 = 20 ,n2 = 20):
        # n1 表示在n1的时间长度上做纯净化 ,n2 表示对20天做移动平均
        try:
            _Turn_slice = ff.rolling_window(self.Turn,n1)
            _PLUS_slice = ff.rolling_window(self.PLUS,n1)
            _len = len(self.Turn) - n1 + 1
            _Turn_cleaned = np.array([np.nan]*len(self.Turn))
            _PLUS_cleaned = np.array([np.nan]*len(self.Turn))                    
            for i in range(_len):
                _a,_b = cleaned(_Turn_slice[i],_PLUS_slice[i])
                _PLUS_cleaned[i+n1-1] = _a[-1]
                _Turn_cleaned[i+n1-1] = _b[-1]
            # n2 表示对20天做移动平均
            _TPS_se,_SPS_se = self.rolling_result(_Turn_cleaned,_PLUS_cleaned,n2)
        except ValueError as e:
            _TPS_se = pd.Series(index = self.index_lst[n2-1:],name = self.name)
            _SPS_se = pd.Series(index = self.index_lst[n2-1:],name = self.name)        
        return _TPS_se,_SPS_se
def get_factor(para):
    name,fre,n,n1,n2 = para
    exam1 = TPS_SPS(name,fre,n)
    _factor = exam1.factor_initial(n2)
    return _factor
    
def get_factor_time_cleaned(para):
    name,fre,n,n1,n2 = para
    exam1 = TPS_SPS(name,fre,n)
    _factor_TPS,_factor_SPS = exam1.factor_time_cleaned(n1,n2)
    _Turn = pd.Series(exam1.Turn,index = exam1.index_lst,name = name)
    _PLUS = pd.Series(exam1.PLUS,index = exam1.index_lst,name = name)
    return _factor_TPS,_factor_SPS,_Turn,_PLUS

def section_cleaned(date_Turn,date_PLUS):
    _PLUS_section_cleaned = np.full_like(date_Turn, np.nan)
    _Turn_section_cleaned = np.full_like(date_Turn, np.nan)
    for i in range(len(date_Turn)):
        _PLUS_section_cleaned[i],_Turn_section_cleaned[i] = cleaned(date_Turn[i],date_PLUS[i])
    return _PLUS_section_cleaned,_Turn_section_cleaned

'''
换手率计算方式：v/v.rolling(n1).mean() 
参数解释:
name:货币代码
fre:数据频率
n 为计算Turn时的回望参数
n1 表示在n1的时间长度上做纯净化 
n2 表示对n2天做移动平均
'''
def main():
    n = 15
    n1 = 5
    n2 = 50
    fre = 4*60
    
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)
            
    para_lst = [(name,fre,n,n1,n2) for name in names]
    with Pool(48) as p:
        res_lst_factor_time_cleaned = list(tqdm(p.imap(get_factor_time_cleaned,para_lst),total = len(para_lst)))
    TPS_lst_time_cleaned,SPS_lst_time_cleaned,Turn_lst,PLUS_lst = zip(*res_lst_factor_time_cleaned)

    TPS_factors_time_cleaned = pd.concat(TPS_lst_time_cleaned,axis = 1).sort_index()
    SPS_factors_time_cleaned = pd.concat(SPS_lst_time_cleaned,axis = 1).sort_index()
    Turn = pd.concat(Turn_lst,axis = 1).sort_index()
    PLUS = pd.concat(PLUS_lst,axis = 1).sort_index()
    PLUS_section_cleaned,Turn_section_cleaned = section_cleaned(Turn.values,PLUS.values)
    
    PLUS_deTurn_df = pd.DataFrame(PLUS_section_cleaned,columns = Turn.columns,index = Turn.index)
    Turn_dePLUS_df = pd.DataFrame(Turn_section_cleaned,columns = Turn.columns,index = Turn.index)
    
    PLUS_deTurn_df_n = PLUS_deTurn_df.rolling(n2,axis=0,min_periods=1).mean()
    Turn_dePLUS_df_n = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).mean()
    STR_dePLUS = Turn_dePLUS_df.rolling(n2,axis=0,min_periods=1).std()
    # 非负处理
    PLUS_deTurn_df_n=PLUS_deTurn_df_n.sub(PLUS_deTurn_df_n.min(axis=1), axis=0)
    Turn_dePLUS_df_n=Turn_dePLUS_df_n.sub(Turn_dePLUS_df_n.min(axis=1), axis=0)
    STR_dePLUS=STR_dePLUS.sub(STR_dePLUS.min(axis=1), axis=0)
    
    TPS = Turn_dePLUS_df_n*PLUS_deTurn_df_n
    SPS = STR_dePLUS*PLUS_deTurn_df_n
    ff.save('TPS_time_RC',TPS_factors_time_cleaned.shift(1,axis=0))
    ff.save('SPS_time_RC',SPS_factors_time_cleaned.shift(1,axis=0))
    ff.save('TPS_section_RC',TPS.shift(1,axis=0))
    ff.save('SPS_section_RC',SPS.shift(1,axis=0))
if __name__ == '__main__':
    main()