import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import os
# 月均耀眼波动率”因子和“月稳耀眼波动率
# 月均耀眼收益率”与“月稳耀眼收益率
class Shining(object):
    def __init__(self,name,k=1,num = 5):
        self.name = name
        self.k =k
        self.num = num
        _v_ret = ff.read_binance(name)[['v','ret']]
        _v_ret.index = pd.to_datetime(_v_ret.index)
        _v_ret_re = _v_ret.resample('1T').asfreq()
        # 为了便于数据处理，只考虑完整的日期
        _start_date = _v_ret_re.index[0].date()  
        _end_date = _v_ret_re.index[-1].date()  
        _keep_start_date = _start_date + pd.Timedelta(days=1)  
        _cleaned_v_ret = _v_ret_re.loc[_keep_start_date:_end_date].iloc[:-1]

        self.volmune = _cleaned_v_ret.v.values
        self.index_lst = np.unique(_cleaned_v_ret.index.date)
        self.ret = _cleaned_v_ret.ret.values
        self.rasing_time, self.raising_time_shift= self.raise_time() # n * 1440
        
    def raise_time(self):
        _delta_vlomune = self.volmune - np.concatenate(([np.nan], self.volmune[:-1]))
        _len = 60*24
        _num = len(_delta_vlomune) // _len
        _delta_vlomune_re = np.reshape(_delta_vlomune,(_num,_len)) # n*1440
        _delta_vlomune_day_mean = np.nanmean(_delta_vlomune_re,axis = 1) # n
        _delta_vlomune_day_std = np.nanstd(_delta_vlomune_re,axis = 1) # n
        _delta_vlomune_day_to = _delta_vlomune_day_mean + self.k*_delta_vlomune_day_std # n
        _delta_vlomune_to = np.repeat(_delta_vlomune_day_to, _len) # (n*1440)
        _raising_time = (self.volmune > _delta_vlomune_to)
        _raising_time_day = np.reshape(_raising_time,(_num,_len))
        # 由于取后五分钟，向后移动四位
        nan_array = np.full((_num, self.num-1), False)
        _raising_time_shift = np.concatenate((nan_array, _raising_time_day[:,self.num-1:]), axis=1)
        return _raising_time_day,_raising_time_shift 

    def shining_factor(self,n):
        nan_array = np.full((self.num-1, self.num), np.nan)
        _ret_slice = np.concatenate((nan_array, ff.rolling_window(self.ret,self.num)), axis=0)
        _ret_day_slice = np.reshape(_ret_slice,(-1,24*60,self.num))
        _ret_day_slice[~self.raising_time_shift] = np.nan # n * 1440 * 5
        # 日耀眼波动率
        _ret_day_slice_std = np.nanstd(_ret_day_slice,axis = 2) # n * 1440 
        _ret_day_std = np.nanmean(_ret_day_slice_std,axis = 1) # n
        # 日耀眼收益率
        _ret_re = np.reshape(self.ret,(-1,24*60)) # n * 1440
        _ret_re_raising = _ret_re * self.rasing_time # n * 1440 
        _ret_day_ret = np.nanmean(_ret_re_raising,axis = 1) # n
        
        nan_array_mon = np.full(n-1, np.nan)
        # 月耀眼波动率
        try:
            _ret_day_std_slice = ff.rolling_window(_ret_day_std,n) # N-n+1*n
            _ret_day_std_slice_mean = np.nanmean(_ret_day_std_slice,axis = 1) # N-n+1
            _ret_day_std_slice_std = np.nanstd(_ret_day_std_slice,axis = 1) # N-n+1
            _ret_day_std_slice_to = (_ret_day_std_slice_mean + _ret_day_std_slice_std)*0.5 # N-n+1
            # 月耀眼收益率
            _ret_day_ret_slice = ff.rolling_window(_ret_day_ret,n) # N-n+1*n
            _ret_day_ret_slice_mean = np.nanmean(_ret_day_ret_slice,axis = 1) # N-n+1
            _ret_day_ret_slice_std = np.nanstd(_ret_day_ret_slice,axis = 1) # N-n+1
            _ret_day_ret_slice_to = (_ret_day_ret_slice_mean + _ret_day_ret_slice_std)*0.5 # N-n+1
            # 等权合成
            adventure_fac = (_ret_day_std_slice_to+_ret_day_ret_slice_to)*0.5
            adventure_final = np.concatenate((nan_array_mon, adventure_fac), axis=0)
        except ValueError as e:
            adventure_final = [np.nan]*len(self.index_lst)
        adventure_se = pd.Series(adventure_final,index = self.index_lst,name = self.name)
        return adventure_se

def get_factor(para):
    name,k,num,n = para
    shining = Shining(name,k = k,num = num)
    _factor = shining.shining_factor(n)
    return _factor
    
def main():
    folder_path = '/home/wangs/data/ba/'
    names = []
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            file_name = os.path.splitext(file)[0] 
            names.append(file_name)
    k=2
    num=3
    n=5
    para_lst = [(name,k,num,n) for name in names]
    with Pool(24) as p:
        res_lst = list(p.imap(get_factor,para_lst))
    advanture_factors = pd.concat(res_lst,axis = 1).sort_index()
    ff.save('advanture_RC',advanture_factors)

if __name__ == '__main__':
    main()