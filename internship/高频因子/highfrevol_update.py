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
来源: 基础因子研究（十四）：高频因子（九），高频波动中的时间序列信息-20201012-长江证券-22页
作者: RC
构造方法:
1. 构建思路：构建分钟级别的高频波动因子，频率可调（1min,5min,15min,30min)
2. 高频波动因子：使用每天的分钟 K 数据计算日内标准差, 取 20 个交易日标准差/均值。（成交量(T) 成交笔数 振幅）
3. 差分高频波动因子（成交量差分标准差因子）：diff_i = vol_i - vol_{i-1}; diff_std_{day} = std{diff_i}/mean{vol_i}; diff_std_{vol} = mean{diff_std_{day}}
4. 绝对值差分高频波动因子（成交量差分绝对值均值因子）：diff_i = vol_i - vol_{i-1}; diff_mean_{day} = mean{abs(diff_i)}/mean{vol_i}; diff_mean_{vol} = mean{diff_mean_{day}}
5. 波峰计数因子：
i、计算全部 240 根K线数据的均值与标准差, 选出大于(均值 +1 倍标准差)的数据; 
ii、统计时间差超过 1 分钟的个数作为当日因子值
'''
# 公用函数
def get_vol_fre_data(stock_name,start,end,fre = 1):
    _stock_vol = ff.read_min(stock_name).loc[start:end,'volume']
    _stock_vol.index = pd.to_datetime(_stock_vol.index).strftime('%Y%m%d')
    if fre == 1:
        _stock_vol.name = stock_name
        result = _stock_vol
    else:
        _total_len = int(len(_stock_vol)/fre)
        _inday_len = int(240/fre)
        _stock_vol_slice = np.reshape(_stock_vol.values,(-1,fre))
        _stock_vol_fre = np.nansum(_stock_vol_slice,axis = -1)
        _index = _stock_vol.index[::fre]
        result = pd.Series(_stock_vol_fre,index = _index,name = stock_name)
    return result

# 因子函数
# 高频波动因子
def get_high_fre_factor_sub(_stock_fre_vol,_date_lst,_fre,_N,_stock_name):
    _inday_num = int(240/_fre)
    _stock_fre_vol_array_slice = np.reshape(_stock_fre_vol.values,(-1,_inday_num))
    _stock_inday_array = np.nanstd(_stock_fre_vol_array_slice,axis = -1) # day
    _stock_inday_array_rolling = ff.rolling_window(_stock_inday_array,_N) # day-N +1,N
    _stock_inday_array_std = np.nanstd(_stock_inday_array_rolling,axis = -1) # day-N +1
    _stock_inday_array_mean = np.nanmean(_stock_inday_array_rolling,axis = -1) # day-N +1
    _stock_inday_factor_array = _stock_inday_array_std/_stock_inday_array_mean # day-N +1
    _index_lst = _stock_fre_vol.index[::_inday_num] # day
    _factor_df = pd.Series(_stock_inday_factor_array,index = _index_lst[(_N-1):],name = _stock_name)    
    result = _factor_df.reindex(_date_lst)
    return result

# 差分高频波动因子
def get_diff_vol_high_fre_factor_sub(_stock_fre_vol,_date_lst,_fre,_N,_stock_name):
    _inday_num = int(240/_fre)
    _stock_fre_vol_array_slice = np.reshape(_stock_fre_vol.values,(-1,_inday_num))
    _stock_fre_diff_array_slice = _stock_fre_vol_array_slice[:,1:] - _stock_fre_vol_array_slice[:,:-1]
    _stock_inday_diff_std_array = np.nanstd(_stock_fre_diff_array_slice,axis = -1) # day
    _stock_inday_vol_mean_array = np.nanmean(_stock_fre_vol_array_slice,axis = -1) # day
    _stock_inday_array = _stock_inday_diff_std_array/_stock_inday_vol_mean_array
    # diff_std_{vol} = mean{diff_std_{day}}
    _stock_inday_array_rolling = ff.rolling_window(_stock_inday_array,_N) # day-N +1,N
    _stock_inday_array_mean = np.nanmean(_stock_inday_array_rolling,axis = -1) # day-N +1
    _index_lst = _stock_fre_vol.index[::_inday_num] # day
    _factor_df = pd.Series(_stock_inday_array_mean,index = _index_lst[(_N-1):],name = _stock_name)
    result = _factor_df.reindex(_date_lst)
    return result

# 绝对值差分高频波动因子
def get_abs_diff_vol_high_fre_factor_sub(_stock_fre_vol,_date_lst,_fre,_N,_stock_name):
    _inday_num = int(240/_fre)
    _stock_fre_vol_array_slice = np.reshape(_stock_fre_vol.values,(-1,_inday_num))
    _stock_fre_abs_diff_array_slice = abs(_stock_fre_vol_array_slice[:,1:] - _stock_fre_vol_array_slice[:,:-1])
    _stock_inday_abs_diff_mean_array = np.nanmean(_stock_fre_abs_diff_array_slice,axis = -1) # day
    _stock_inday_vol_mean_array = np.nanmean(_stock_fre_vol_array_slice,axis = -1) # day
    _stock_inday_array = _stock_inday_abs_diff_mean_array/_stock_inday_vol_mean_array
    # diff_mean_{vol} = mean{diff_mean_{day}}
    _stock_inday_array_rolling = ff.rolling_window(_stock_inday_array,_N) # day-N +1,N
    _stock_inday_array_mean = np.nanmean(_stock_inday_array_rolling,axis = -1) # day-N +1
    _index_lst = _stock_fre_vol.index[::_inday_num] # day
    _factor_df = pd.Series(_stock_inday_array_mean,index = _index_lst[(_N-1):],name = _stock_name)
    result = _factor_df.reindex(_date_lst)
    return result

# 波峰计数因子
def get_peak_count_factor_sub(_stock_fre_vol,_date_lst,_fre,_N,_stock_name):
    _inday_num = int(240/_fre)
    _stock_fre_vol_array_slice = np.reshape(_stock_fre_vol.values,(-1,_inday_num)) # day，inday
    _stock_inday_mean = np.nanmean(_stock_fre_vol_array_slice,axis = -1) # day
    _stock_inday_std = np.nanstd(_stock_fre_vol_array_slice,axis = -1) # day
    _select_option = _stock_inday_mean+_stock_inday_std
    _select_result = np.full(_stock_fre_vol_array_slice.shape,0)
    _stock_fre_vol_array_slice_match = np.where(_stock_fre_vol_array_slice > _select_option[:,np.newaxis],1,_select_result) # day，inday
    _stock_fre_vol_array_slice_match_diff = _stock_fre_vol_array_slice_match[:,1:] - _stock_fre_vol_array_slice_match[:,:-1]
    _stock_array_slice_num_peak = np.where(_stock_fre_vol_array_slice_match_diff != 1,0,_stock_fre_vol_array_slice_match_diff) # day，inday-1
    _factor_array = np.nansum(_stock_array_slice_num_peak,axis = -1)
    _index_lst = _stock_fre_vol.index[::_inday_num] # day
    _factor_df = pd.Series(_factor_array,index = _index_lst,name = _stock_name)
    result = _factor_df.reindex(_date_lst)
    return result

def get_all_factor_sub(para):
    _stock_name,_date_lst,_fre_lst,_N = para
    _start = _date_lst[0][:4] + '-' +_date_lst[0][4:6] + '-'+_date_lst[0][6:] 
    _end_date = str(int(_date_lst[-1]) + 1)
    _end = _end_date[:4] + '-' +_end_date[4:6] + '-'+_end_date[6:] 
    _stock_vol = {}
    _all_factor = []
    _option = True
    for i in range(4):  
        _fre = _fre_lst[i]
        _vol_name = f"_stock_vol{_fre}"
        if _vol_name not in _stock_vol:
            try:
                _stock_vol[_vol_name] = get_vol_fre_data(_stock_name,_start,_end,_fre)
                if len(_stock_vol[_vol_name]) < _N*240/_fre:
                    _stock_factor = pd.Series(index = _date_lst,name = _stock_name)  
                    _option = False
            except:
                _stock_factor = pd.Series(index = _date_lst,name = _stock_name)  
                _option = False
        if _option:
            if i == 0:
                _stock_factor = get_high_fre_factor_sub(_stock_vol[_vol_name],_date_lst,_fre,_N,_stock_name)
            elif i == 1:
                _stock_factor = get_diff_vol_high_fre_factor_sub(_stock_vol[_vol_name],_date_lst,_fre,_N,_stock_name)
            elif i == 2:
                _stock_factor = get_abs_diff_vol_high_fre_factor_sub(_stock_vol[_vol_name],_date_lst,_fre,_N,_stock_name)
            elif i == 3:
                _stock_factor = get_peak_count_factor_sub(_stock_vol[_vol_name],_date_lst,_fre,_N,_stock_name)
            
        _all_factor.append(_stock_factor)
    return _all_factor[0],_all_factor[1],_all_factor[2],_all_factor[3]
    
def get_all_factor(date_lst,fre0,fre1,fre2,fre3,N = 20):
    _stock_lst = ff.filter0.index
    fre_lst = [fre0,fre1,fre2,fre3]
    para_lst = [(_stock_name,date_lst,fre_lst,N) for _stock_name in _stock_lst]
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(get_all_factor_sub,para_lst),total = len(_stock_lst)))
    _factor0_lst,_factor1_lst,_factor2_lst,_factor3_lst = zip(*res_lst)
    _factor0 = pd.concat(_factor0_lst,axis = 1).sort_index().T
    _factor1 = pd.concat(_factor1_lst,axis = 1).sort_index().T
    _factor2 = pd.concat(_factor2_lst,axis = 1).sort_index().T
    _factor3 = pd.concat(_factor3_lst,axis = 1).sort_index().T
    return _factor0,_factor1,-_factor2,-_factor3


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
    N = 20
    date_lst = ff.filter0.columns
    fre1 = 1
    fre2 = 1
    fre3 = 1
    fre0 = 1
    update_win = 20
    date_lst = date_lst[-(N+update_win+10):] 
    high_fre_vol_RC,high_fre_diff_vol_RC,high_fre_absdiff_vol_RC,peak_count_vol_RC = get_all_factor(date_lst,fre0,fre1,fre2,fre3,N)
    # update
    his_high_fre_vol_RC = ff.read('high_fre_vol_RC').to_dict()
    new_high_fre_vol_RC = (high_fre_vol_RC.shift(1,axis = 1) * ff.filter0).iloc[:,-update_win:].to_dict()
    his_high_fre_vol_RC.update(new_high_fre_vol_RC)
    his_high_fre_vol_RC = pd.DataFrame(his_high_fre_vol_RC)

    his_high_fre_diff_vol_RC = ff.read('high_fre_diff_vol_RC').to_dict()
    new_high_fre_diff_vol_RC = (high_fre_diff_vol_RC.shift(1,axis = 1) * ff.filter0).iloc[:,-update_win:].to_dict()
    his_high_fre_diff_vol_RC.update(new_high_fre_diff_vol_RC)
    his_high_fre_diff_vol_RC = pd.DataFrame(his_high_fre_diff_vol_RC)

    his_high_fre_absdiff_vol_RC = ff.read('high_fre_absdiff_vol_RC').to_dict()
    new_high_fre_absdiff_vol_RC = (high_fre_absdiff_vol_RC.shift(1,axis = 1) * ff.filter0).iloc[:,-update_win:].to_dict()
    his_high_fre_absdiff_vol_RC.update(new_high_fre_absdiff_vol_RC)
    his_high_fre_absdiff_vol_RC = pd.DataFrame(his_high_fre_absdiff_vol_RC)

    his_peak_count_vol_RC = ff.read('peak_count_vol_RC').to_dict()
    new_peak_count_vol_RC = (peak_count_vol_RC.shift(1,axis = 1) * ff.filter0).iloc[:,-update_win:].to_dict()
    his_peak_count_vol_RC.update(new_peak_count_vol_RC)
    his_peak_count_vol_RC = pd.DataFrame(his_peak_count_vol_RC)
    # check
    assert check(ff.read('high_fre_vol_RC'),his_high_fre_vol_RC)
    assert check(ff.read('high_fre_diff_vol_RC'),his_high_fre_diff_vol_RC)
    assert check(ff.read('high_fre_absdiff_vol_RC'),his_high_fre_absdiff_vol_RC)
    assert check(ff.read('peak_count_vol_RC'),his_peak_count_vol_RC)
    
    version_reserve(his_high_fre_vol_RC,'high_fre_vol_RC') # 旧版本保留
    ff.save('high_fre_vol_RC',his_high_fre_vol_RC)
    
    version_reserve(his_high_fre_diff_vol_RC,'high_fre_diff_vol_RC') # 旧版本保留
    ff.save('high_fre_diff_vol_RC',his_high_fre_diff_vol_RC)
    
    version_reserve(his_high_fre_absdiff_vol_RC,'high_fre_absdiff_vol_RC') # 旧版本保留
    ff.save('high_fre_absdiff_vol_RC',his_high_fre_absdiff_vol_RC)
    
    version_reserve(his_peak_count_vol_RC,'peak_count_vol_RC') # 旧版本保留
    ff.save('peak_count_vol_RC',his_peak_count_vol_RC)
    
if __name__ == '__main__':
    main()