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
6. 股价自相关性的风险度量因子：我们将假设价格序列存在自相关性，即股
价波动并不随机的情况下，通过 正态分布刻画日内的个股 VaR ，15 日标准差作为
最终因子，叫做 rtn_condVaR7
4“崩盘”概率因子数首先基于前一个交易日的分钟级收益率序列，计算得到的连续上涨/下跌次数的样本数据，并进一步计算得到对于连续下跌泊松分布中参数𝜆的估计；计算个股连续下跌和连续上涨的差异：𝑥为全市场当日连续上涨𝜆𝑡𝑝𝑜𝑠中位数，𝑥 + 𝑘为当日连续下跌𝜆𝑡𝑛𝑒𝑔前25%分位数；最终引入泊松分布的累计分布函数，计算得到“崩盘”概率。计算个股过去 15 日“崩盘”概率的标准差，记为 flashCrashProb 因子。08
5等权合成上述七个因子，记为时序信息复合因子。相乘
'''
def get_data(para):
    name,fre = para
    data =  pd.DataFrame()
    ini_data = ff.read_binance(name).loc[:'20240216 235900']
    ini_data.index = pd.to_datetime(ini_data.index)
    v_day = ini_data.v.resample('D').sum()
    v_fre = ini_data.v.resample(f'{fre}T').sum()
    data.loc[:,'v_fra'] = v_fre/v_day[v_fre.index.date].values
    data.loc[:,'ret'] = ini_data.ret.resample(f'{fre}T').apply(lambda x: (x+1).prod()-1)
    return data
    
def get_data_simple(para):
    name,fre = para
    ini_data = ff.read_binance(name).ret.loc[:'20240216 235900']
    ini_data.index = pd.to_datetime(ini_data.index)
    if fre == 1:
        ret = ini_data.resample('1T').asfreq()
        ret.name = name
    else:
        ret = ini_data.resample(f'{fre}T').apply(lambda x: (x+1).prod()-1)
        ret.name = name
    return ret

def get_data_simple2(para):
    name,fre = para
    ini_data = ff.read_binance(name).c.loc[:'20240216 235900']
    ini_data.index = pd.to_datetime(ini_data.index)
    if fre == 1:
        close = ini_data.resample('1T').asfreq()
        close.name = name
    else:
        close = ini_data.resample(f'{fre}T').last()
        close.name = name
    return close

def get_foc(para):
    data = get_data(para)
    rtn_foc = data.ret.resample('4H').apply(lambda x: x.autocorr())
    vol_foc = data.v_fra.resample('4H').apply(lambda x: x.autocorr())
    foc_Comb = (rtn_foc+vol_foc)*0.5
    rtn_foc.name = para[0]
    vol_foc.name = para[0]
    foc_Comb.name = para[0]
    return rtn_foc,vol_foc,foc_Comb

def dw(data):
    result = (data.diff()**2).sum()/((data-data.mean())**2).sum()
    return result

def get_dw(para):
    data = get_data(para)
    rtn_dw = data.ret.resample('4H').apply(lambda x: dw(x))
    vol_dw = data.v_fra.resample('4H').apply(lambda x: dw(x))
    dw_Comb = (rtn_dw+vol_dw)*0.5
    rtn_dw.name = para[0]
    vol_dw.name = para[0]
    dw_Comb.name = para[0]
    return rtn_dw,vol_dw,dw_Comb

def rho(data):
    data_s1 = np.array([np.nan] + list(data[:-1]))
    valid_indices = np.logical_and(~np.isnan(data), ~np.isnan(data_s1))
    data_valid = data[valid_indices]
    data_s1_valid = data_s1[valid_indices]
    data_res = np.nan
    # 防止所有位置均为空值的情况
    if not np.all(~valid_indices):
        data_s1_c = sm.add_constant(data_s1_valid)
        data_model = sm.OLS(data_valid, data_s1_c)
        data_results = data_model.fit()
        try:
            data_res = data_results.params[1]
        except:
            pass  
    return data_res

'''
参数设定：
Q:统计量计算：最高到n=5阶自相关性
回望区间从3*n(15)到6*n(30)
'''
def get_Q(data):
    h = 5
    Q = np.nan
    if len(data) >= h:
        acf = sm.tsa.acf(data, nlags=h)
        n = len(data)
        Q = n * (n + 2) * sum(acf[k]**2 / (n - k) for k in range(1, h + 1))
    return Q

def get_highStdRtn_mean(data):
    data_std = np.array([np.std(data[max(i - 6 + 1, 0):i+1]) for i in range(len(data))])
    percentile_80 = np.percentile(data_std, 80)
    data_above_80_percentile = data[data_std > percentile_80]
    return np.mean(data_above_80_percentile)

def get_VaR(data):
    mu = np.mean(data)
    sigma = np.std(data)
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

def get_prob(data):
    poisson_dist = poisson(mu=data[0]*10)
    prob = poisson_dist.cdf(data[2]*10)-poisson_dist.cdf(data[1]*10)
    return prob

# 截面标准化
def scale(data):
    scaled_data = (data-np.mean(data))/np.std(data)
    return scaled_data


def main():
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)

    fre = 20
    get_data_para_lst = [(name,fre) for name in names]
    with Pool(16) as p:
        foc_res_lst = list(tqdm(p.imap(get_foc, get_data_para_lst), total=len(get_data_para_lst)))  
        dw_res_lst = list(tqdm(p.imap(get_dw, get_data_para_lst), total=len(get_data_para_lst)))  
        get_data_res_lst = list(tqdm(p.imap(get_data, get_data_para_lst), total=len(get_data_para_lst)))  
    # foc
    foc_rtn_list,foc_vol_list,foc_Comb_list = zip(*foc_res_lst)
    foc_Comb = pd.concat(foc_Comb_list,axis=1).sort_index()
    foc_Comb = foc_Comb.rolling(window = 15,min_periods = 1).mean()
    # dw
    DW_rtn_list,DW_vol_list,DW_Comb_list = zip(*dw_res_lst)
    DW_Comb = pd.concat(DW_Comb_list,axis=1).sort_index()
    DW_Comb = DW_Comb.rolling(window = 15,min_periods = 1).mean()
    # rho
    v_fra = pd.DataFrame([res['v_fra'] for res in get_data_res_lst], index=names).T.sort_index()
    ret = pd.DataFrame([res['ret'] for res in get_data_res_lst], index=names).T.sort_index()
    fre_num = int(4*60/fre)
    fre_len = int(v_fra.shape[0]*v_fra.shape[1]/fre_num)
    v_fra_list = np.reshape(v_fra.T.values, (fre_len,fre_num))
    ret_list = np.reshape(ret.T.values, (fre_len,fre_num))

    with Pool(16) as p:
        rho_res_v_fra_lst = list(tqdm(p.imap(rho, v_fra_list), total=len(v_fra_list))) 
        rho_res_ret_lst = list(tqdm(p.imap(rho, ret_list), total=len(ret_list))) 

    day_num = len(np.unique(v_fra.index.date))*6
    name_num = len(names)
    rho_res_v_fra_lst_reshape = np.reshape(rho_res_v_fra_lst, (name_num,day_num))
    rho_res_ret_lst_reshape = np.reshape(rho_res_ret_lst, (name_num,day_num))
    rtn_rho = pd.DataFrame(rho_res_ret_lst_reshape,index = names,columns = DW_Comb.index).T
    vol_rho = pd.DataFrame(rho_res_v_fra_lst_reshape,index = names,columns = DW_Comb.index).T
    rho_Comb = (rtn_rho+vol_rho)*0.5
    
    rho_Comb = rho_Comb.rolling(window = 15,min_periods = 1).std()
    # LBQ
    fre = 5
    get_data_para_lst = [(name,fre) for name in names]
    with Pool(34) as p:
        get_data_res_lst = list(tqdm(p.imap(get_data, get_data_para_lst), total=len(get_data_para_lst)))  
    v_fra_5 = pd.DataFrame([res['v_fra'] for res in get_data_res_lst], index=names).T.sort_index()
    ret_5 = pd.DataFrame([res['ret'] for res in get_data_res_lst], index=names).T.sort_index()
    fre_len= int(4*60/fre)
    fre_num = int(ret_5.shape[0]*ret_5.shape[1]/fre_len)
    v_fra_5_list = np.reshape(v_fra_5.T.values,(fre_num,fre_len))
    ret_5_list = np.reshape(ret_5.T.values,(fre_num,fre_len))
    v_fra_5_lst = []
    ret_5_lst = []
    for i in range(24):
        v_fra_5_lst+=list(v_fra_5_list[:,-(6+i):])
        ret_5_lst+=list(ret_5_list[:,-(6+i):])
    
    with Pool(32) as p:
        v_fra_5_Q_period_lst = list(tqdm(p.imap(get_Q, v_fra_5_lst), total=len(v_fra_5_lst))) 
        ret_5_Q_period_lst = list(tqdm(p.imap(get_Q, ret_5_lst), total=len(ret_5_lst))) 
    fre_len1= int(len(v_fra_5_Q_period_lst)/24)
    fre_num1 = int(24)
    rtn_LBQ_all_data= np.reshape(ret_5_Q_period_lst,(fre_num1,fre_len1)).T
    vol_LBQ_all_data= np.reshape(v_fra_5_Q_period_lst,(fre_num1,fre_len1)).T
    rtn_LBQ_std_data = np.std(rtn_LBQ_all_data, axis=1)
    vol_LBQ_std_data = np.std(vol_LBQ_all_data, axis=1)
    fre_len2= int(len(rtn_LBQ_std_data)/len(names))
    fre_num2 = int(len(names))
    rtn_LBQ_lst = np.reshape(rtn_LBQ_std_data,(fre_num2,fre_len2))
    vol_LBQ_lst = np.reshape(vol_LBQ_std_data,(fre_num2,fre_len2))
    rtn_LBQ = pd.DataFrame(rtn_LBQ_lst,index = names ,columns =DW_Comb.index).T
    vol_LBQ = pd.DataFrame(vol_LBQ_lst,index = names ,columns =DW_Comb.index).T
    LBQ_Comb = (rtn_LBQ+vol_LBQ)*0.5
    LBQ_Comb = LBQ_Comb.rolling(window = 15,min_periods = 1).mean()

    # highStdRtn_mean_lst
    ret_list = np.reshape(ret_5.T.values,(fre_num,fre_len))
    
    with Pool(24) as p:
        highStdRtn_mean_lst = list(tqdm(p.imap(get_highStdRtn_mean, ret_list), total=len(ret_list))) 
    
    day_num = len(np.unique(v_fra.index.date))*6
    name_num = len(names)
    highStdRtn_mean_lst_reshape = np.reshape(highStdRtn_mean_lst, (name_num,day_num))
    
    highStdRtn_mean = pd.DataFrame(highStdRtn_mean_lst_reshape,columns =DW_Comb.index,index = names).T

    # VaR
    fre = 1
    get_data_para_lst = [(name,fre) for name in names]
    with Pool(34) as p:
        get_data_res_lst = list(tqdm(p.imap(get_data_simple2, get_data_para_lst), total=len(get_data_para_lst)))  
        flashCrashProb_get_data_res_lst = list(tqdm(p.imap(get_data_simple, get_data_para_lst), total=len(get_data_para_lst)))  
    close_1 = pd.concat(get_data_res_lst,axis = 1).sort_index()
    fre_num = int(4*60/fre)
    fre_len = int(close_1.shape[0]*close_1.shape[1]/fre_num)
    close_list = np.reshape(close_1.T.values, (fre_len,fre_num))
    
    with Pool(24) as p:
        VaR_lst = list(tqdm(p.imap(get_VaR, close_list), total=len(close_list))) 
    
    day_num = len(np.unique(close_1.index.date))*6
    name_num = len(names)
    VaR_lst_reshape = np.reshape(VaR_lst, (name_num,day_num))
    
    VaR = pd.DataFrame(VaR_lst_reshape,columns = DW_Comb.index,index = names).T
    VaR = VaR.rolling(window = 15,min_periods = 1).std()
    # flashCrashProb
    ret_1 = pd.concat(flashCrashProb_get_data_res_lst,axis = 1).sort_index()
    fre_num = int(4*60)
    fre_len = int(ret_1.shape[0]*ret_1.shape[1]/fre_num)
    ret_list = np.reshape(ret_1.T.values, (fre_len,fre_num))
    
    with Pool(16) as p:
        lambda_lst = list(tqdm(p.imap(get_lambda, ret_list), total=len(ret_list))) 
    pos_lambda_lst ,neg_lambda_lst = zip(*lambda_lst)
    pos_lambda_lst_reshape = np.reshape(pos_lambda_lst, (name_num,day_num))
    neg_lambda_lst_reshape = np.reshape(neg_lambda_lst, (name_num,day_num))
    pos_lambda = pd.DataFrame(pos_lambda_lst_reshape,columns = DW_Comb.index ,index = names).T
    neg_lambda = pd.DataFrame(neg_lambda_lst_reshape,columns = DW_Comb.index ,index = names).T

    ne_lambda_lst = neg_lambda.T.values.flatten().tolist()
    x_k = neg_lambda.quantile(0.75,axis = 1).values.tolist()*len(names)
    x = pos_lambda.quantile(0.5,axis = 1).values.tolist()*len(names)
    data_lst = ne_lambda_lst+x+x_k
    fre_num = int(3)
    fre_len = int(len(data_lst)/fre_num)
    data_lst_reshape = np.reshape(data_lst,(fre_num,fre_len)).T
    with Pool(24) as p:
        prob_list = list(tqdm(p.imap(get_prob, data_lst_reshape), total=len(data_lst_reshape))) 
    
    fre_num = int(len(names))
    fre_len = int(len(prob_list)/fre_num)
    prob_list_reshape = np.reshape(prob_list,(fre_num,fre_len))
        
    prob_df = pd.DataFrame(prob_list_reshape,index = names,columns = DW_Comb.index).T
    flashCrashProb = prob_df.rolling(window =15,min_periods=1).std()

    # 等权合成
    para_lst = foc_Comb.values.tolist() + DW_Comb.values.tolist() + rho_Comb.values.tolist() + LBQ_Comb.values.tolist() + highStdRtn_mean.values.tolist() + VaR.values.tolist() + flashCrashProb.values.tolist() 
    with Pool(24) as p:
        final_res_lst = list(tqdm(p.imap(scale,para_lst),total = len(para_lst)))
    _num = int(7)
    _len = int(len(para_lst)*len(names)/7)
    final_res_lst_reshape = np.reshape(final_res_lst,(_num,_len))
    foc_Comb_scale_lst = final_res_lst_reshape[0] #升序
    DW_Comb_scale_lst = final_res_lst_reshape[1] #降序
    rho_Comb_scale_lst = final_res_lst_reshape[2] #升序
    LBQ_Comb_scale_lst = final_res_lst_reshape[3] #降序
    highStdRtn_mean_scale_lst = final_res_lst_reshape[4] #升序
    VaR_scale_lst = final_res_lst_reshape[5] #降序
    flashCrashProb_scale_lst = final_res_lst_reshape[6] #降序
    fre_num = int(len(foc_Comb_scale_lst)/len(names))
    fre_len = int(len(names))
    foc_Comb_scale_lst_reshape = np.reshape(foc_Comb_scale_lst,(fre_num,fre_len))
    foc_Comb_scale = pd.DataFrame(foc_Comb_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    DW_Comb_scale_lst_reshape = np.reshape(DW_Comb_scale_lst,(fre_num,fre_len))
    DW_Comb_scale = pd.DataFrame(DW_Comb_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    rho_Comb_scale_lst_reshape = np.reshape(rho_Comb_scale_lst,(fre_num,fre_len))
    rho_Comb_scale = pd.DataFrame(rho_Comb_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    LBQ_Comb_scale_lst_reshape = np.reshape(LBQ_Comb_scale_lst,(fre_num,fre_len))
    LBQ_Comb_scale = pd.DataFrame(LBQ_Comb_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    highStdRtn_mean_scale_lst_reshape = np.reshape(highStdRtn_mean_scale_lst,(fre_num,fre_len))
    highStdRtn_mean_scale = pd.DataFrame(highStdRtn_mean_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    VaR_scale_lst_reshape = np.reshape(VaR_scale_lst,(fre_num,fre_len))
    VaR_scale = pd.DataFrame(VaR_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    flashCrashProb_scale_lst_reshape = np.reshape(flashCrashProb_scale_lst,(fre_num,fre_len))
    flashCrashProb_scale = pd.DataFrame(flashCrashProb_scale_lst_reshape,index = foc_Comb.index,columns = foc_Comb.columns)
    final_factor = (foc_Comb_scale - DW_Comb_scale + rho_Comb_scale - LBQ_Comb_scale + highStdRtn_mean_scale - VaR_scale - flashCrashProb_scale)/7

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