import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib as mpl
mpl.rc("font", family='Droid Sans Fallback', weight="bold")
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import statsmodels.api as sm
from scipy import stats
from scipy.stats import poisson
import os
'''
名称: foc_Comb_RC,DW_Comb_RC,rho_Comb_RC,LBQ_Comb_RC,highStdRtn_mean_RC,VaR_RC,flashCrashProb_RC,final_factor_RC
来源: 20230629-兴业证券-高频研究系列六：时序信息中的Alpha
作者: RC
构造方法:
1. 自相关系数因子：rtn_foc 分钟收益率一阶自相关系数；vol_foc 分钟成交量占比一阶自相关系数；过去15 日指标的均值作为最终的因子值；foc_Comb 等权合成 rtn_foc 和 vol_foc 因子
2. D-W 统计量因子：rtn_DW 分钟收益率 D-W 统计量；vol_DW 分钟成交量占比 D-W 统计量；过去15 日指标的均值作为最终的因子值；DW_Comb 等权合成 rtn_DW 和 vol_DW 因子
3. 残差自相关系数因子： rtn_rho 分钟收益率残差自相关系数；vol_rho 分钟成交量残差自相关系数；取 15 日指标的标准差作为最终因子值；rho_Comb 等权合成 rtn_rho 和 vol_rho 因子
4. 非同步交易性因子：对于同一天而言，根据分钟收益率和分钟成交量占比计算不同回望区间下的 Q 统计量序列，计算其每日标准差并取时序上 15 日均值tn_LBQ 与 vol_LBQ等权合成 LBQ_Comb
5. 高波收益率均值因子：计算各个分钟节点过去 30分钟的 5 分钟滚动收益率标准差，并筛选出标准差处于日内该股 80%分位数以上的时间节点，并统计该时间节点中 5 分钟滚动收益率的均值，最终构建因子highStdRtn_mean。
6. 股价自相关性的风险度量因子：我们将假设价格序列存在自相关性，即股价波动并不随机的情况下，通过二正态分布刻画日内的个股 VaR 值，15 日标准差作为最终因子，叫做 rtn_condVaR
7. “崩盘”概率因子，首先基于前一个交易日的分钟级收益率序列，计算得到的连续上涨/下跌次数的样本数据，并进一步计算得到对于连续下跌泊松分布中参数𝜆的估计；计算个股连续下跌和连续上涨的差异：𝑥为全市场当日连续上涨𝜆𝑡𝑝𝑜𝑠中位数，𝑥 + 𝑘为当日连续下跌𝜆𝑡𝑛𝑒𝑔前25%分位数；最终引入泊松分布的累计分布函数，计算得到“崩盘”概率。计算个股过去 15 日“崩盘”概率的标准差，记为 flashCrashProb 因子。
8. 等权合成上述七个因子，记为时序信息复合因子。
'''
def get_corr(_data):
    _data_s1 = np.array([np.nan] + list(_data[:-1]))
    _valid_indices = np.logical_and(~np.isnan(_data), ~np.isnan(_data_s1))
    _data_valid = _data[_valid_indices]
    _data_s1_valid = _data_s1[_valid_indices]
    _data_res = np.nan
    # 防止所有位置均为空值的情况
    if not np.all(~_valid_indices):
        _data_res = np.corrcoef(_data_valid,_data_s1_valid)[0,1]
    return _data_res

def get_res_rho(_data):
    _data_s1 = np.array([np.nan] + list(_data[:-1]))
    _valid_indices = np.logical_and(~np.isnan(_data), ~np.isnan(_data_s1))
    _data_valid = _data[_valid_indices]
    _data_s1_valid = _data_s1[_valid_indices]
    _data_res = np.nan
    # 防止所有位置均为空值的情况
    if not np.all(~_valid_indices):
        _data_s1_c = sm.add_constant(_data_s1_valid)
        _data_model = sm.OLS(_data_valid, _data_s1_c)
        _data_results = _data_model.fit()
        _data_res = get_corr(_data_results.resid)
    return _data_res

def get_VaR(data):
    mu = np.nanmean(data)
    sigma = np.nanstd(data)
    data1 = np.array([np.nan] + list(data[:-1]))
    valid_indices = np.logical_and(~np.isnan(data), ~np.isnan(data1))
    rho = np.corrcoef(data[valid_indices], data1[valid_indices])[0][1]
    Sigma = rho*np.array([[1,rho],[rho,1]])
    C = np.array([[1,-1],[0,1]])
    nu_Sigma = np.dot(np.dot(C.T,Sigma),C)
    ad_mu = nu_Sigma[0][1]/nu_Sigma[1][1]*(data[-1]-mu)
    ad_sigma = nu_Sigma[0][0] - nu_Sigma[0][1]/nu_Sigma[1][1]*nu_Sigma[1][0]
    VaR = ad_mu - 1.96*ad_sigma**0.5
    return VaR
        
def get_lambda(data):
    pos_lam = np.nan
    neg_lam = np.nan
    pos_lst = []
    neg_lst = []
    m = 0
    n = 0
    j = 0
    for i in data:
        if i <0 and j <0:
            n += 1
        elif i >0 and j >0:
            m += 1
        elif m != 0:
            pos_lst.append(m)
            m = 0
        elif n != 0:
            neg_lst.append(n)
            n = 0
        j = i
    if m!=0:
        pos_lst.append(m)
        m = 0   
    if n!=0:
        neg_lst.append(n)
        n = 0     
    if len(pos_lst) > 0:
        pos_lam = np.mean(pos_lst)
    if len(neg_lst) > 0:
        neg_lam = np.mean(neg_lst)    
    return pos_lam,neg_lam


class Composite(object):
    def __init__(self,fre,name,n = 15,h = 5,min_ret = 5,back_length = 30): # n为前n日平均,h为Q统计量计算时的最高阶,min_ret为高波收益率计算时ret的长度，back_length为回看时间段
        # fre为最终计算出的因子频率
        self.name = name
        self.fre =fre
        self.n =n
        self.h =h
        _v_c_ret = ff.read_binance(name)[['v','c','ret']]
        _v_c_ret.index = pd.to_datetime(_v_c_ret.index)
        _v_c_ret_re = _v_c_ret.resample('1T').asfreq()
        # 为了便于数据处理，只考虑完整的日期
        _start_date = _v_c_ret_re.index[0].date()  
        _end_date = _v_c_ret_re.index[-1].date()  
        _keep_start_date = _start_date + pd.Timedelta(days=1)  
        _cleaned_v_c_ret = _v_c_ret_re.loc[_keep_start_date:_end_date].iloc[:-1]
        self.len_ =len(_cleaned_v_c_ret.index)
        self.close_day = np.reshape(_cleaned_v_c_ret.c.values,(-1,fre))
        self.index_lst = _cleaned_v_c_ret.index[::fre]
        self._option = True
        if len(self.index_lst) < n:
            self._option = False
        self.ret_day = np.reshape(_cleaned_v_c_ret.ret.values,(-1,fre))
        self.vol_day = np.reshape(self.vol_fra(_cleaned_v_c_ret.v.values),(-1,fre))
        _c_5 = _cleaned_v_c_ret.c.resample(f'{min_ret}T').last()
        self.back_length = back_length
        self.min_ret = min_ret
        self.ret_day_ad = np.reshape((_c_5/_c_5.shift(1) - 1).values,(-1,fre//min_ret)) 

    def vol_fra(self,vol):
        _vol_day =  np.nansum(vol.reshape(self.len_//self.fre,self.fre),axis = 1)
        _vol_day_re = np.repeat(_vol_day, self.fre)
        _vol_fra = vol/_vol_day_re
        return _vol_fra
        
    def factor_foc(self):
        _ret_day_foc = np.array([np.nan]*(self.len_//self.fre))
        _vol_day_foc = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _ret_day_foc[i] = get_corr(self.ret_day[i,:])
            _vol_day_foc[i] = get_corr(self.vol_day[i,:])  
        _foc_Comb = (_ret_day_foc+_vol_day_foc)*0.5
        _foc_Comb_slice = ff.rolling_window(_foc_Comb,self.n)
        _foc_Comb_mean = np.nanmean(_foc_Comb_slice,axis = 1)
        return _foc_Comb_mean

    def factor_DW(self):
        _ret_day_DW = np.array([np.nan]*(self.len_//self.fre))
        _vol_day_DW = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _ret_day_DW[i] = np.nansum((self.ret_day[i,1:] - self.ret_day[i,:-1])**2)/np.nansum((self.ret_day[i,:] - np.nanmean(self.ret_day[i,:]))**2)
            _vol_day_DW[i] = np.nansum((self.vol_day[i,1:] - self.vol_day[i,:-1])**2)/np.nansum((self.vol_day[i,:] - np.nanmean(self.vol_day[i,:]))**2)
        _DW_Comb = (_ret_day_DW+_vol_day_DW)*0.5
        _DW_Comb_slice = ff.rolling_window(_DW_Comb,self.n)
        _DW_Comb_mean = np.nanmean(_DW_Comb_slice,axis = 1)
        return _DW_Comb_mean   
        
    def factor_rho(self):
        _ret_day_rho = np.array([np.nan]*(self.len_//self.fre))
        _vol_day_rho = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _ret_day_rho[i] = get_res_rho(self.ret_day[i,:])
            _vol_day_rho[i] = get_res_rho(self.vol_day[i,:])
        _ret_day_rho_slice = ff.rolling_window(_ret_day_rho,self.n)
        _vol_day_rho_slice = ff.rolling_window(_vol_day_rho,self.n)
        _ret_day_rho_std = np.nanstd(_ret_day_rho_slice,axis = 1)
        _vol_day_rho_std = np.nanstd(_vol_day_rho_slice,axis = 1)
        _rho_Comb = (_ret_day_rho_std + _vol_day_rho_std) * 0.5
        return _rho_Comb

    '''
    参数设定：
    Q:统计量计算：最高到n=5阶自相关性
    回望区间从3*n(15)到6*n(30)
    '''
    def get_Q(self,data):
        h = self.h
        Q = np.nan
        if len(data) >= h:
            acf = sm.tsa.acf(data, nlags=h)
            n = len(data)
            Q = n * (n + 2) * sum(acf[k]**2 / (n - k) for k in range(1, h + 1))
        return Q

    def factor_LBQ(self):
        # 取每日回望[15,20,30,45,60,90,120]的区间计算Q序列
        _back_period = [15,20,30,45,60,90,120]
        _ret_day_LBQ = np.array([np.nan]*(self.len_//self.fre))
        _vol_day_LBQ = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _ret_LBQ_inday = []
            _vol_LBQ_inday = []
            for _k in _back_period:
                _ret_LBQ_inday.append(self.get_Q(self.ret_day[i,-_k:]))
                _vol_LBQ_inday.append(self.get_Q(self.vol_day[i,-_k:]))
            _ret_day_LBQ[i] = np.nanstd(np.array(_ret_LBQ_inday))
            _vol_day_LBQ[i] = np.nanstd(np.array(_vol_LBQ_inday))
        _LBQ_Comb = (_ret_day_LBQ+_vol_day_LBQ)*0.5
        _LBQ_Comb_slice = ff.rolling_window(_LBQ_Comb,self.n)
        _LBQ_Comb_mean = np.nanmean(_LBQ_Comb_slice,axis = 1)
        return _LBQ_Comb_mean   

    def factor_highStdRtn_mean(self,percent = 80):
        _highStdRtn_mean = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _ret_inday_ad_slice = ff.rolling_window(self.ret_day_ad[i,:],self.back_length//self.min_ret)
            _ret_inday_ad_std = np.nanstd(_ret_inday_ad_slice,axis=1)
            _percentile = np.percentile(_ret_inday_ad_std, percent)
            _ret_inday_ad_above_percentile = self.ret_day_ad[i,(self.back_length//self.min_ret - 1):][_ret_inday_ad_std > _percentile]
            _highStdRtn_mean[i] = np.nanmean(_ret_inday_ad_above_percentile)
        return _highStdRtn_mean[(self.n-1):]

    def factor_rtn_condVaR(self):
        _rtn_condVaR = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _rtn_condVaR[i] = get_VaR(self.close_day[i,:])
        _rtn_condVaR_slice = ff.rolling_window(_rtn_condVaR,self.n)
        _rtn_condVaR_std = np.nanstd(_rtn_condVaR_slice,axis = 1)   
        return _rtn_condVaR_std

    def factor_flashCrashProb(self):
        _pos_day_lam = np.array([np.nan]*(self.len_//self.fre))
        _neg_day_lam = np.array([np.nan]*(self.len_//self.fre))
        _flashCrashProb = np.array([np.nan]*(self.len_//self.fre))
        for i in range(self.len_//self.fre):
            _pos_day_lam[i], _neg_day_lam[i] = get_lambda(self.ret_day[i,:])
            # 做截面比较是没有意义的，不妨将10lambda_pos作为poisson分布参数，计算累计分布概率至10lambda_neg
            _poisson_dist = poisson(mu=_pos_day_lam[i]*10)
            _flashCrashProb[i] = _poisson_dist.cdf(_neg_day_lam[i]*10) - _poisson_dist.cdf(_pos_day_lam[i]*10)
        _flashCrashProb_slice = ff.rolling_window(_flashCrashProb,self.n)
        _flashCrashProb_std = np.nanstd(_flashCrashProb_slice,axis = 1)   
        return _flashCrashProb_std        

    def factor_final(self):
        if self._option:
            _factor_foc = self.factor_foc()
            _factor_DW = -self.factor_DW()
            _factor_rho = self.factor_rho()
            _factor_LBQ = -self.factor_LBQ()
            _factor_highStdRtn_mean = self.factor_highStdRtn_mean()
            _factor_rtn_condVaR = -self.factor_rtn_condVaR()
            _factor_flashCrashProb = -self.factor_flashCrashProb()
            _factor_final_com = np.vstack([_factor_foc,_factor_DW,_factor_rho,_factor_LBQ,_factor_highStdRtn_mean,_factor_rtn_condVaR,_factor_flashCrashProb]).T
            _factor_final = np.array([np.nan]*(self.len_//self.fre - self.n + 1))
            for i in range(self.len_//self.fre - self.n + 1):
                _scaled_data = (_factor_final_com[i,:]-np.nanmean(_factor_final_com[i,:]))/np.nanstd(_factor_final_com[i,:])
                _factor_final[i] = np.nanmean(_scaled_data)
            _factor_foc_se = pd.Series(_factor_foc,index = self.index_lst[self.n -1:],name = self.name)
            _factor_DW_se = pd.Series(_factor_DW,index = self.index_lst[self.n -1:],name = self.name)
            _factor_rho_se = pd.Series(_factor_rho,index = self.index_lst[self.n -1:],name = self.name)
            _factor_LBQ_se = pd.Series(_factor_LBQ,index = self.index_lst[self.n -1:],name = self.name)
            _factor_highStdRtn_mean_se = pd.Series(_factor_highStdRtn_mean,index = self.index_lst[self.n -1:],name = self.name)
            _factor_rtn_condVaR_se = pd.Series(_factor_rtn_condVaR,index = self.index_lst[self.n -1:],name = self.name)
            _factor_flashCrashProb_se = pd.Series(_factor_flashCrashProb,index = self.index_lst[self.n -1:],name = self.name)
            _factor_final_se = pd.Series(_factor_final,index = self.index_lst[self.n -1:],name = self.name)
        else:
            _factor_foc_se = pd.Series(name = self.name)
            _factor_DW_se = pd.Series(name = self.name)
            _factor_rho_se = pd.Series(name = self.name)
            _factor_LBQ_se = pd.Series(name = self.name)
            _factor_highStdRtn_mean_se = pd.Series(name = self.name)
            _factor_rtn_condVaR_se = pd.Series(name = self.name)
            _factor_flashCrashProb_se = pd.Series(name = self.name)
            _factor_final_se = pd.Series(name = self.name)
        return _factor_foc_se,_factor_DW_se,_factor_rho_se,_factor_LBQ_se,_factor_highStdRtn_mean_se,_factor_rtn_condVaR_se,_factor_flashCrashProb_se,_factor_final_se

def get_result(para):
    fre ,name = para
    _example = Composite(fre,name)
    return _example.factor_final()
    
def main():
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)
    # n为前n日平均,h为Q统计量计算时的最高阶,min_ret为高波收益率计算时ret的长度，back_length为回看时间段
    # fre为最终计算出的因子频率
    fre = 24*60
    para_lst = [(fre,name) for name in names]
    res_lst = []
    for para in tqdm(para_lst):
        res_lst.append(get_result(para))
    factor_foc_se,_factor_DW_se,_factor_rho_se,_factor_LBQ_se,_factor_highStdRtn_mean_se,_factor_rtn_condVaR_se,_factor_flashCrashProb_se,_factor_final_se = zip(*res_lst)
    foc_Comb = pd.concat(factor_foc_se,axis=1).sort_index()
    DW_Comb = pd.concat(_factor_DW_se,axis=1).sort_index()
    rho_Comb = pd.concat(_factor_rho_se,axis=1).sort_index()
    LBQ_Comb = pd.concat(_factor_LBQ_se,axis=1).sort_index()
    highStdRtn_mean = pd.concat(_factor_highStdRtn_mean_se,axis=1).sort_index()
    VaR = pd.concat(_factor_rtn_condVaR_se,axis=1).sort_index()
    flashCrashProb = pd.concat(_factor_flashCrashProb_se,axis=1).sort_index()
    final_factor = pd.concat(_factor_final_se,axis=1).sort_index()
    ff.save('foc_Comb_RC',foc_Comb)
    ff.save('DW_Comb_RC',DW_Comb)
    ff.save('rho_Comb_RC',rho_Comb)
    ff.save('LBQ_Comb_RC',LBQ_Comb)
    ff.save('highStdRtn_mean_RC',highStdRtn_mean)
    ff.save('VaR_RC',VaR)
    ff.save('flashCrashProb_RC',flashCrashProb)
    ff.save('final_factor_RC',final_factor)
if __name__ == '__main__':
    main()