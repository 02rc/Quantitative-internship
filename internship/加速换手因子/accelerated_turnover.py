import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import extend
'''
名称: accelerated_turnover_rank_RC
来源: 20240316-华安证券-量化研究系列报告之十五：加速换手因子，“适逢其时”的换手奥秘
作者: RC
构造方法:
1. 构建辅助因子：i、日内收益率，日内收益率的构建方法为，今日收盘价减去今日开盘价的差除以昨日收盘价；ii、振荡幅度，振幅的构建方法为，今日最高价减去今日最低价的差除以昨日收盘价；iii:价格因子，影线差。构建方法为上影线减去下影线后除以昨日收盘价。
2. 用截面数据，换手率作为因变量，影线差 PLUS 作为自变量，使用一元线性回归的方法取得残差项。为Turn_dePLUS，20天移动平均得到：Turn_dePLUS20；
3. Turn_dePLUS 因子值回滚20天取标准差为STR_dePLUS
4. 用截面数据，PLUS 因子作为因变量，换手率作为自变量，进行横截面的一元线性回归，取得残差项，为 PLUS_deTurn。20天移动平均得到：PLUS_deTurn20；
5. TPS 全名为 Turn20 conformed by PLUS，构建方法是把 Turn20_dePLUS 与PLUS_deTurn20 相乘；SPS 全名为 STR conformed by PLUS，构建方法是把 STR_dePLUS 与 PLUS_deTurn20相乘
'''

start, end = '20170104', '20230418'
sw_lst=['申万-国防军工I', '申万-采掘I', '申万-家用电器I', '申万-公用事业I', '申万-通信I', '申万-农林牧渔I', '申万-食品饮料I', '申万-计算机I', '申万-有色金属I', '申万-机械设备I', '申万-交通运输I', '申万-建筑材料I', '申万-银行I', '申万-纺织服装I', '申万-房地产I', '申万-传媒I', '申万-商业贸易I', '申万-综合I', '申万-汽车I', '申万-轻工制造I', '申万-建筑装饰I', '申万-电子I', '申万-非银金融I', '申万-化工I', '申万-休闲服务I', '申万-医药生物I', '申万-电气设备I', '申万-钢铁I','申万-煤炭I', '申万-环保I', '申万-石油石化I', '申万-美容护理I']

def read_data(name,start_day=start,end_day=end):
    if name in ('open', 'close', 'high', 'low'):
        result = (ff.read(name) * ff.read('post') * ff.filter0).loc[ff.filter0.index, start_day:end_day]
    else:
        result = (ff.read(name) * ff.filter0).loc[ff.filter0.index, start_day:end_day]
    return result

def get_event_sub(data_df,N,type):
    if type == '放量' or '上涨' or '活跃' or '高振幅':
        _rolling_data = ff.rolling_window(data_df,N)
        _N_mean = np.nanmean(_rolling_data,axis = -1) # stock ,N_0 - N + 1 
        result_array = np.where(data_df.values[:,(N-1):]>_N_mean,1,np.nan) # stock ,N_0 - N + 1 
    elif type == '缩量' or '下跌' or '低振幅':
        _rolling_data = ff.rolling_window(data_df,N)
        _N_mean = np.nanmean(_rolling_data,axis = -1) # stock ,N_0 - N + 1 
        result_array = np.where(data_df.values[:,(N-1):]<_N_mean,1,np.nan) # stock ,N_0 - N + 1 
    return result_array
    
def get_industry_mean_sub(industry):
    ind = read_data(industry,'factor')
    indus_amount=data_df*ind
    indus_amount_sum=(indus_amount.isna().replace({True:np.nan,False:1}))*(indus_amount.mean())    
    return indus_amount_sum
    
def get_industry_mean(data_array):
    result_lst = []
    for industry in tqdm(sw_lst):
        industry_array = read_data(industry).values
        industry_data_array = industry_array * data_array
        industry_array_mean = np.nanmean(industry_data_array,axis = 0)
        industry_array_mean_all = industry_array_mean*industry_array
        result_lst.append(industry_array_mean_all)
    result_array_all = np.array(result_lst)
    result_array = np.nansum(result_array_all,axis = 0)
    return result_array
    
def get_factor_sub(para):
    # 当日市场或行业成分股加速换手平均值的差值;先使用市场（好算）；再使用行业;算了，都用吧
    _array,_type2 = para
    _array_acc_turn, _array_indus_mean_acc_turn = _array # N_date - N2 + 1,N2
    if _type2 == '平均':
        _res_array = _array_acc_turn*(_array_indus_mean_acc_turn - np.nanmean(_array_indus_mean_acc_turn,axis = 1)[:, np.newaxis])/np.nanstd(_array_indus_mean_acc_turn,axis = 1)[:, np.newaxis]
    elif _type2 == '最小':
        _res_array = _array_acc_turn*(_array_indus_mean_acc_turn - np.nanmin(_array_indus_mean_acc_turn,axis = 1)[:, np.newaxis])/np.nanstd(_array_indus_mean_acc_turn,axis = 1)[:, np.newaxis]
    return _res_array # N_date - N2 + 1,N2
          
class acc_turn(object):
    def __init__(self,start_day,end_day,n = 20):
        self.start = start_day
        self.end = end_day
        _turn = read_data('turnover_rate',start_day,end_day)
        _acc_turn_df = _turn/_turn.shift(1,axis = 1) - 1
        _standered_acc_turn_df = _acc_turn_df.rolling(n,axis = 1).std()
        self.date_lst = _acc_turn_df.columns # N_date
        self.stock_lst = _acc_turn_df.index # N_stock
        self.acc_turn = _acc_turn_df.values # N_stock,N_date
        self.standered_acc_turn = (_acc_turn_df/_standered_acc_turn_df).values # N_stock,N_date
        self.indus_mean_acc_turn = get_industry_mean(self.acc_turn) # N_stock,N_date
        self.indus_mean_standered_acc_turn = get_industry_mean(self.standered_acc_turn) # N_stock,N_date
        self.all_mean_acc_turn = _acc_turn_df.mean().values # N_date
        self.all_mean_standered_acc_turn = _standered_acc_turn_df.mean().values # N_date
        
    def get_factor(self,type1,type2,type3,N1 = 3,N2 = 20):
        # type1: 放量上涨 or 高振幅
        # type2: 最小 or 平均
        # type3: 是否平稳，True or False

        # 事件区域部分
        if type1 == ['放量','上涨']:
            if not hasattr(self, 'rising_heavy_volume_event'):
                if not hasattr(self, 'heavy_volume_event'):
                    _vol = read_data('vol',self.start,self.end)
                    self.heavy_volume_event = get_event_sub(_vol,N1,'放量') # N_stock,N_date - N1 + 1 
                if not hasattr(self, 'rising_event'):
                    _close = read_data('close',self.start,self.end)
                    self.rising_event = get_event_sub(_close,N1,'上涨') # N_stock,N_date - N1 + 1 
                self.rising_heavy_volume_event = self.heavy_volume_event * self.rising_event# N_stock,N_date - N1 + 1 
            _event = ff.rolling_window(self.rising_heavy_volume_event,N2) # N_stock,N_date - N1 - N2 + 2,N2
        elif type1 == '高振幅':
            if not hasattr(self, 'high_amplitude_event'):
                _close = read_data('close',self.start,self.end)
                _high = read_data('high',self.start,self.end)
                _low = read_data('low',self.start,self.end)
                _TR = (_high - _low)/_close.shift(1,axis = 1)
                self.high_amplitude_event = get_event_sub(_TR,N1,'高振幅') # N_stock,N_date - N1 + 1 
            _event = ff.rolling_window(self.high_amplitude_event,N2) # N_stock,N_date - N1 - N2 + 2,N2

        # 因子计算部分
        # 平稳
        if type3:
            _rolling_standered_acc_turn = ff.rolling_window(self.standered_acc_turn,N2) # N_stock,N_date - N2 + 1,N2
            _rolling_indus_mean_standered_acc_turn = ff.rolling_window(self.indus_mean_standered_acc_turn,N2) # N_stock,N_date - N2 + 1,N2
            para_lst = [((_rolling_standered_acc_turn[i,:,:],_rolling_indus_mean_standered_acc_turn[i,:,:]),type2) for i in range(len(self.stock_lst))] # N_stock
            if type2 == '最小':
                if not hasattr(self, 'min_standered_factor_before'):
                    with Pool(24) as p:
                        _res_lst = list(tqdm(p.imap(get_factor_sub,para_lst),total = len(self.stock_lst))) # N_stock,N_date - N2 + 1,N2
                    self.min_standered_factor_before = np.array(_res_lst) # N_stock,N_date - N2 + 1,N2
                _factor_without_event = self.min_standered_factor_before[:,(N1-1):,:] # N_stock,N_date - N1 - N2 + 2,N2
            elif type2 == '平均':
                if not hasattr(self, 'mean_standered_factor_before'):
                    with Pool(24) as p:
                        _res_lst = list(tqdm(p.imap(get_factor_sub,para_lst),total = len(self.stock_lst))) # N_stock,N_date - N2 + 1,N2
                    self.mean_standered_factor_before = np.array(_res_lst) # N_stock,N_date - N2 + 1,N2
                _factor_without_event = self.mean_standered_factor_before[:,(N1-1):,:] # N_stock,N_date - N1 - N2 + 2,N2     
        # 非平稳
        if not type3:
            _rolling_acc_turn = ff.rolling_window(self.acc_turn,N2) # N_stock,N_date - N2 + 1,N2
            _rolling_indus_acc_turn = ff.rolling_window(self.indus_mean_acc_turn,N2) # N_stock,N_date - N2 + 1,N2
            para_lst = [((_rolling_acc_turn[i,:,:],_rolling_indus_acc_turn[i,:,:]),type2) for i in range(len(self.stock_lst))] # N_stock
            if type2 == '最小':
                if not hasattr(self, 'min_factor_before'):
                    with Pool(24) as p:
                        _res_lst = list(tqdm(p.imap(get_factor_sub,para_lst),total = len(self.stock_lst))) # N_stock,N_date - N2 + 1,N2
                    self.min_factor_before = np.array(_res_lst) # N_stock,N_date - N2 + 1,N2
                _factor_without_event = self.min_factor_before[:,(N1-1):,:] # N_stock,N_date - N1 - N2 + 2,N2
            elif type2 == '平均':
                if not hasattr(self, 'mean_factor_before'):
                    with Pool(24) as p:
                        _res_lst = list(tqdm(p.imap(get_factor_sub,para_lst),total = len(self.stock_lst))) # N_stock,N_date - N2 + 1,N2
                    self.mean_factor_before = np.array(_res_lst) # N_stock,N_date - N2 + 1,N2
                _factor_without_event = self.mean_factor_before[:,(N1-1):,:] # N_stock,N_date - N1 - N2 + 2,N2

        # 组合部分
        _factor_array = np.nansum(_event * _factor_without_event,axis = -1) # N_stock,N_date - N1 - N2 + 2
        _factor_array[_factor_array == 0] = np.nan
        _result = pd.DataFrame(_factor_array,columns = self.date_lst[N1+N2-2:],index = self.stock_lst)
        return _result       

# 对称正交后进行等权合成
def symmetric(matrix):
    _len = matrix.shape[1]
    result = np.array([np.nan] * _len)
    df = pd.DataFrame(matrix.T)
    df_c = df.dropna()
    index_c = df_c.index
    F = np.array(df_c,dtype=np.float64)
    # 求$M_{K*K}$的特征值和特征向量
    Lambda,Q = np.linalg.eigh(np.dot(F.T, F))
    D = np.diag(Lambda ** (-0.5))
    D = np.where(np.isnan(D), 0, D)
    Fhat = np.dot(F,Q)
    Fhat = np.dot(Fhat,D)
    Fhat = np.dot(Fhat,Q.T)
    for i in range(Fhat.shape[1]):
        Fhat[:,i] = Fhat[:,i] / np.sqrt(np.dot(Fhat[:,i], Fhat[:,i]))
    sum_Fhat = np.nansum(Fhat,axis = 1)
    result[index_c] = sum_Fhat
    return result # N_stock

def get_final_factor(n = 20, N1 = 3, N2 = 20,start = '20170104', end = '20230418'):
    final_example = acc_turn(start, end, n)
    min_rising_heavy_volume_factor = final_example.get_factor(['放量','上涨'],'最小',False,N1,N2) # 放量上涨日最小加速换手偏离
    mean_rising_heavy_volume_factor = final_example.get_factor(['放量','上涨'],'平均',False,N1,N2) # 放量上涨日平均加速换手偏离
    min_standered_rising_heavy_volume_factor = final_example.get_factor(['放量','上涨'],'最小',True,N1,N2) # 放量上涨日最小平稳加速换手偏离
    mean_standered_rising_heavy_volume_factor = final_example.get_factor(['放量','上涨'],'平均',True,N1,N2) # 放量上涨日平均平稳加速换手偏离
    min_high_amplitude_factor = final_example.get_factor('高振幅','最小',False,N1,N2) # 高振幅日最小加速换手偏离
    mean_high_amplitude_factor = final_example.get_factor('高振幅','平均',False,N1,N2) # 高振幅日平均加速换手偏离
    min_standered_high_amplitude_factor = final_example.get_factor('高振幅','最小',True,N1,N2) # 高振幅日最小平稳加速换手偏离
    mean_standered_high_amplitude_factor = final_example.get_factor('高振幅','平均',True,N1,N2) # 高振幅日平均平稳加速换手偏离
    date_lst = final_example.date_lst[N1+N2-2:]
    stock_lst = final_example.stock_lst
    matrix_lst = [np.stack((min_rising_heavy_volume_factor.values[:,i], mean_rising_heavy_volume_factor.values[:,i],min_standered_rising_heavy_volume_factor.values[:,i],mean_standered_rising_heavy_volume_factor.values[:,i],min_high_amplitude_factor.values[:,i],mean_high_amplitude_factor.values[:,i],min_standered_high_amplitude_factor.values[:,i],mean_standered_high_amplitude_factor.values[:,i]), axis=0) for i in range(len(date_lst))]
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(symmetric,matrix_lst),total = len(date_lst))) # N_date_lst,N_stock
    final_factor = pd.DataFrame(res_lst,columns = stock_lst,index = date_lst).T
    return final_factor, min_rising_heavy_volume_factor, mean_rising_heavy_volume_factor, min_standered_rising_heavy_volume_factor, mean_standered_rising_heavy_volume_factor, min_high_amplitude_factor, mean_high_amplitude_factor, min_standered_high_amplitude_factor, mean_standered_high_amplitude_factor

def main():
    start, end = '20170104', '20230418'
    final_factor, min_rising_heavy_volume_factor, mean_rising_heavy_volume_factor, min_standered_rising_heavy_volume_factor, mean_standered_rising_heavy_volume_factor, min_high_amplitude_factor, mean_high_amplitude_factor, min_standered_high_amplitude_factor, mean_standered_high_amplitude_factor = get_final_factor()
    final_factor_rank = min_rising_heavy_volume_factor.rank() - mean_rising_heavy_volume_factor.rank() + min_standered_rising_heavy_volume_factor.rank() - mean_standered_rising_heavy_volume_factor.rank() + min_high_amplitude_factor.rank() - mean_high_amplitude_factor.rank() + min_standered_high_amplitude_factor.rank() - mean_standered_high_amplitude_factor.rank()
    
    mv = ff.read('total_mv').loc[:,final_factor_rank.columns]
    total_index = set(final_factor_rank.index) & set(mv.index)
    final_factor_rank_neu = extend.spread_reg(final_factor_rank.loc[total_index,:], mv.loc[total_index,:], ind=True) # ind=True为同时进行市值与行业中性化
    ff.save('accelerated_turnover_rank_RC',final_factor_rank_neu.shift(1,axis = 1) * ff.filter0)
if __name__ == '__main__':
    main()