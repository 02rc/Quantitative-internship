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

def get_data_ret(para):
    name ,fre = para
    if fre != 1:
        ini_c = ff.read_binance(name).c.loc[:'20240216 235900']
        ini_c.index = pd.to_datetime(ini_c.index)
        c = ini_c.resample(f'{fre}T').last()
        ret = c/c.shift(1) - 1
        ret.name = name
    else:
        ini_ret = ff.read_binance(name).ret
        ini_ret.index = pd.to_datetime(ini_ret.index)
        ret = ini_ret.resample('1T').asfreq()
        ret.name = name
    return ret

def get_data_sigma(para):
    name ,fre1, fre2 = para
    if fre2 == 1:
        ini_ret = ff.read_binance(name).ret.loc[:'20240216 235900']
        ini_ret.index = pd.to_datetime(ini_ret.index)
        sigma = (ini_ret**2).resample(f'{fre1}T').mean()
        sigma.name = name
    else:
        ini_c = ff.read_binance(name).c.loc[:'20240216 235900']
        ini_c.index = pd.to_datetime(ini_c.index)
        c = ini_c.resample(f'{fre2}T').last()
        ret = c/c.shift(1) - 1    
        sigma = (ret**2).resample(f'{fre1}T').mean()
        sigma.name = name
    return sigma

def get_pearson(data):
    data0 = np.reshape(data,(2,int(len(data)/2)))
    data1 = data0[0]
    data2 = data0[1]
    invaid_index = np.logical_and(~np.isnan(data1), ~np.isnan(data2))
    data1_invaid = data1[invaid_index]
    data2_invaid = data2[invaid_index]
    correlation_matrix = np.corrcoef(data1_invaid, data2_invaid)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def main():
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)
    # 4H一天
    fre_total = 4*60
    # 日间
    para_lst1 = [(name,fre_total) for name in names]
    fre = 1
    para_lst2 = [(name,fre_total,fre) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index()
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    # 001
    day_index = sigma_day.index
    day_index_se = sigma_day.iloc[:,0]
    # 001
    window = int(20)
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)[:,:-1]
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)[:,1:]
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lag_day_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lag_day_lst)/arr_num)
    vol_lag_day_reshape = np.reshape(vol_lag_day_lst,(arr_num,arr_len))
    col = ret_day.index[window:]
    vol_lag_day = pd.DataFrame(vol_lag_day_reshape,columns = col, index = names).T
    # 日间2
    fre = 5
    para_lst2 = [(name,fre_total,fre) for name in names]
    with Pool(34) as p:
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)[:,:-1]
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)[:,1:]
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lag_day2_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lag_day2_lst)/arr_num)
    vol_lag_day2_reshape = np.reshape(vol_lag_day2_lst,(arr_num,arr_len))
    col = ret_day.index[window:]
    vol_lag_day2 = pd.DataFrame(vol_lag_day2_reshape,columns = col, index = names).T
    # 日内高频
    fre1 = 1
    fre2 = 5
    para_lst1 = [(name,fre2) for name in names]
    para_lst2 = [(name,fre2,fre1) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index()
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    arr_len = int(fre_total/fre2)
    arr_num = int(ret_day.shape[0]*ret_day.shape[1]/arr_len)
    ret_day_arr = np.reshape(ret_day.T.values,(arr_num,arr_len))[:,:-1]
    sigma_day_arr = np.reshape(sigma_day.T.values,(arr_num,arr_len))[:,1:]
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lag_inday_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lag_inday_lst)/arr_num)
    vol_lag_inday_reshape = np.reshape(vol_lag_inday_lst,(arr_num,arr_len))
    col = day_index
    vol_lag_inday = pd.DataFrame(vol_lag_inday_reshape,columns = col, index = names).T

    # 日间
    para_lst1 = [(name,fre_total) for name in names]
    fre1 = 1
    fre2 = int(fre_total*10)
    para_lst2 = [(name,fre2,fre1) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index().resample(f'{fre_total}T').ffill()
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index().loc[sigma_day.index,:]
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)[:,1:]
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)[:,:-1]
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lead_day_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lead_day_lst)/arr_num)
    vol_lead_day_reshape = np.reshape(vol_lead_day_lst,(arr_num,arr_len))
    col = ret_day.index[window:]
    vol_lead_day = pd.DataFrame(vol_lead_day_reshape,columns = col, index = names).T
    # 日间2
    fre = 5
    para_lst2 = [(name,fre2,fre) for name in names]
    with Pool(34) as p:
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index().resample(f'{fre_total}T').ffill()
    
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)[:,1:]
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)[:,:-1]
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lead_day2_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lead_day2_lst)/arr_num)
    vol_lead_day2_reshape = np.reshape(vol_lead_day2_lst,(arr_num,arr_len))
    col = ret_day.index[window:]
    vol_lead_day2 = pd.DataFrame(vol_lead_day2_reshape,columns = col, index = names).T
    # 日内高频
    fre1 = 1
    fre2 = 5
    fre3 = 60
    para_lst1 = [(name,fre2) for name in names]
    para_lst2 = [(name,fre3,fre1) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index().resample(f'{fre2}T').ffill()
    index_last = day_index_se[:sigma_day.index[-1]].index[-1]
    sigma_day = sigma_day.loc[:index_last,:].iloc[:-1,:]
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index().loc[:index_last,:].iloc[:-1,:]
    arr_len = int(fre_total/fre2)
    arr_num = int(ret_day.shape[0]*ret_day.shape[1]/arr_len)
    ret_day_arr = np.reshape(ret_day.T.values,(arr_num,arr_len))[:,1:]
    sigma_day_arr = np.reshape(sigma_day.T.values,(arr_num,arr_len))[:,:-1]
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        vol_lead_inday_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(vol_lead_inday_lst)/arr_num)
    vol_lead_inday_reshape = np.reshape(vol_lead_inday_lst,(arr_num,arr_len))
    col = day_index_se[:index_last].iloc[:-1].index
    vol_lead_inday = pd.DataFrame(vol_lead_inday_reshape,columns = col, index = names).T

    # 日间
    fre_total = 4*60
    para_lst1 = [(name,fre_total) for name in names]
    fre = 1
    para_lst2 = [(name,fre_total,fre) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index()
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    # 001
    day_index = sigma_day.index
    day_index_se = sigma_day.iloc[:,0]
    # 001
    window = int(20)
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        con_effect_day_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(con_effect_day_lst)/arr_num)
    con_effect_day_reshape = np.reshape(con_effect_day_lst,(arr_num,arr_len))
    col = ret_day.index[(window-1):]
    con_effect_day = pd.DataFrame(con_effect_day_reshape,columns = col, index = names).T
    # 日间2
    fre = 5
    para_lst2 = [(name,fre_total,fre) for name in names]
    with Pool(34) as p:
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    
    ret_day_arr_name = ff.rolling_window(ret_day.T,window)
    sigma_day_arr_name = ff.rolling_window(sigma_day.T,window)
    ret_day_arr = np.concatenate(ret_day_arr_name, axis=0)
    sigma_day_arr = np.concatenate(sigma_day_arr_name, axis=0)
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        con_effect_day2_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(con_effect_day2_lst)/arr_num)
    con_effect_day2_reshape = np.reshape(con_effect_day2_lst,(arr_num,arr_len))
    col = ret_day.index[(window-1):]
    con_effect_day2 = pd.DataFrame(con_effect_day2_reshape,columns = col, index = names).T
    # 日内高频
    fre1 = 1
    fre2 = 5
    para_lst1 = [(name,fre2) for name in names]
    para_lst2 = [(name,fre2,fre1) for name in names]
    with Pool(34) as p:
        ret_day_lst = list(tqdm(p.imap(get_data_ret,para_lst1),total=len(para_lst1)))
        sigma_day_lst = list(tqdm(p.imap(get_data_sigma,para_lst2),total=len(para_lst2)))
    ret_day = pd.concat(ret_day_lst,axis = 1).sort_index()
    sigma_day = pd.concat(sigma_day_lst,axis = 1).sort_index()
    arr_len = int(fre_total/fre2)
    arr_num = int(ret_day.shape[0]*ret_day.shape[1]/arr_len)
    ret_day_arr = np.reshape(ret_day.T.values,(arr_num,arr_len))
    sigma_day_arr = np.reshape(sigma_day.T.values,(arr_num,arr_len))
    data_arr = np.concatenate((ret_day_arr, sigma_day_arr),axis =1)
    with Pool(24) as p:
        con_effect_inday_lst = list(tqdm(p.imap(get_pearson,data_arr),total=len(data_arr)))
    arr_num = len(names)
    arr_len = int(len(con_effect_inday_lst)/arr_num)
    con_effect_inday_reshape = np.reshape(con_effect_inday_lst,(arr_num,arr_len))
    col = day_index
    con_effect_inday = pd.DataFrame(con_effect_inday_reshape,columns = col, index = names).T

    ff.save('vol_lag_day_RC',vol_lag_day)
    ff.save('vol_lag_day2_RC',vol_lag_day2)
    ff.save('vol_lag_inday_RC',vol_lag_inday)
    ff.save('vol_lead_day_RC',vol_lead_day)
    ff.save('vol_lead_day2_RC',vol_lead_day2)
    ff.save('vol_lead_inday_RC',vol_lead_inday)
    ff.save('con_effect_day_RC',con_effect_day)
    ff.save('con_effect_day2_RC',con_effect_day2)
    ff.save('con_effect_inday_RC',con_effect_inday)

if __name__ == '__main__':
    main()