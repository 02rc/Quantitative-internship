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

'''
名称: overnightsmart20_RC,CTR_RC,jumpCTR_RC 
来源: 20240109-东吴证券-技术分析拥抱选股因子”系列（十五）：换手率切割刀CTR因子Cutlets of TurnoverRate，换手率的异质信念
作者: RC
构造方法:
1. 构建思路：使用其余因子对换手率因子进行切割
2. 成交量对动量因子的修正：日内切割：取股票过去 20 个交易日，使用日内换手率对日内收益率进行切割（5组），求均值，计算高换手因子
3. 成交量对动量因子的修正：隔夜切割：取股票过去 20 个交易日，使用昨日换手率对隔夜收益率进行切割，求均值，计算高换手因子
4. 换手率对异质信念的识别：日内收益率对日内换手率做切割，计算高收益因子
5. 换手率对异质信念的识别：次日隔夜收益率的辅助,利用次日隔夜收益率对日内换手率做切割，计算低收益因子
6. 换手率对异质信念的识别：次日隔夜换手率的辅助,利用次日隔夜换手率对日内换手率做切割，计算高换手因子
7. 次日隔夜聪明钱:OvernightSmart $_t=\frac{\frac{r_t-\min \left(r_1, r_2, \ldots, r_{20}\right)}{\max \left(r_1, r_2 \ldots, r_{20}\right)-\min \left(r_1, r_2, \ldots, r_{20}\right)}}{T_t}$,取过去 20 个交易日的“隔夜聪明钱”均值作为新因子 OvernightSmart20；
8. 换手率切割刀 CTR 因子（Cutlets of Turnover Rate）:按照次日隔夜聪明钱对日内换手率切割，计算低隔夜聪明钱因子；做市值中性化
5. 殊途同归：抢跑 CTR 因子：按照次日隔夜聪明钱从低到高排序；取隔夜聪明钱最小的三天的换手率，加上本月最后一个交易日的日内换手率，取这四天的换手率平均值后市值中性化得到新因子，抢跑 CTR（JumpCTR）。
'''
def read_data(name,start,end):
    if name in ('open', 'close', 'high', 'low'):
        result = (ff.read(name) * ff.read('post') * ff.filter0).loc[:, start:end]
    else:
        result = (ff.read(name) * ff.filter0).loc[:, start:end]
    return result
    
def get_initial_data_sub(para):
    _stock_name ,_start ,_end = para
    try:
        _min_turn = ff.read_min(_stock_name).loc[_start:_end,'volume']
        _min_turn.index = pd.to_datetime(_min_turn.index).strftime('%Y%m%d')
        _day_turn = _min_turn.iloc[::240]
        _day_turn.name = _stock_name
        return _day_turn
    except:
        pass

# 隔夜收益率，隔夜换手率，日内收益率，日内换手率
def get_initial_data(date_list,stock_list):
    _start = date_list[0][:4] + '-' +date_list[0][4:6] + '-'+date_list[0][6:] 
    _end_date = str(int(date_list[-1]) + 1)
    _end = _end_date[:4] + '-' +_end_date[4:6] + '-'+_end_date[6:] 
    # 日内换手率和隔夜换手率
    all_turnover = ff.read('turnover_rate')
    para_lst = [(_stock_name ,_start ,_end) for _stock_name in stock_list]
    with Pool(36) as p:
        bt_vol_lst = list(tqdm(p.imap(get_initial_data_sub,para_lst),total = len(stock_list)))
    cleaned_bt_vol_lst = [item for item in bt_vol_lst if item is not None]
    bt_vol = pd.concat(cleaned_bt_vol_lst,axis = 1).T
    bt_turnover = bt_vol/ff.read('float_share')
    id_turnover = all_turnover - bt_turnover
    bt_turnover = bt_turnover.reindex(index = stock_list,columns = date_list)
    id_turnover = id_turnover.reindex(index = stock_list,columns = date_list)
    # 日内收益率和隔夜收益率
    bt_ret = read_data('open',date_list[0],date_list[-1])/read_data('close',date_list[0],date_list[-1]).shift(1,axis = 1) - 1
    id_ret = ff.rets_all - bt_ret
    bt_ret = bt_ret.reindex(index = stock_list,columns = date_list)
    id_ret = id_ret.reindex(index = stock_list,columns = date_list)    
    return bt_turnover,id_turnover,bt_ret,id_ret

'''
array2d = np.array([
    [3, 1, 2, 9, 7, np.nan],
    [8, 6, 4, 5, 0, np.nan],
    [7, 2, 3, 1, 8, np.nan]
])
partitioned_indices = np.argpartition(array2d, -4, axis=1)[:, -4:]
bool_indices = np.zeros_like(array2d, dtype=bool)
rows = np.arange(array2d.shape[0])[:, None]
bool_indices[rows, partitioned_indices] = True
'''
def cutting_sub1(para):
    array1 ,array2 = para # N_date - N + 1 ,20
    max_index = np.argpartition(array2, 16,axis = -1)[:,:16]
    rows = np.arange(array1.shape[0])[:, None]
    array1[rows, max_index] = np.nan
    result = np.nanmean(array1,axis  = -1)
    return result
    
def cutting_sub2(para):
    array1 ,array2 = para
    min_index = np.argpartition(array2, -16,axis = -1)[:,-16:]
    rows = np.arange(array1.shape[0])[:, None]
    array1[rows, min_index] = np.nan
    result = np.nanmean(array1,axis  = -1)
    return result

def cutting_sub3(para):
    array1 ,array2 = para
    _last = array1[:,-1]
    min_index = np.argpartition(array2, -17,axis = -1)[:,-17:]
    rows = np.arange(array1.shape[0])[:, None]
    array1[rows, min_index] = np.nan
    arr = np.concatenate((array1, _last[:, np.newaxis]), axis=-1)
    result = np.nanmean(arr,axis  = -1)
    return result
    
def cutting(data1,data2,type):
    n_stock,n_date = data1.shape
    # 按照data2对data1切割
    data1_array_rolling = ff.rolling_window(data1,20) # n_stock,n_date - N + 1,N
    data2_array_rolling = ff.rolling_window(data2,20) # n_stock,n_date - N + 1,N
    _num,_,_ = data1_array_rolling.shape
    para_lst = [(data1_array_rolling[i,:,:],data2_array_rolling[i,:,:]) for i in range(_num)]
    if type == 'min':
        with Pool(24) as p:
            res_lst = list(tqdm(p.imap(cutting_sub2,para_lst),total = _num)) # n_stock*n_date - N + 1
    elif type == 'max':
        with Pool(24) as p:
            res_lst = list(tqdm(p.imap(cutting_sub1,para_lst),total = _num))  
    elif type == 'jump':
        with Pool(24) as p:
            res_lst = list(tqdm(p.imap(cutting_sub3,para_lst),total = _num))  
    result_df = pd.DataFrame(res_lst,columns = data1.columns[19:],index = data1.index)
    return result_df

def overnight_smart_money_sub(para):
    overnight_turnover_array,overnight_ret_array = para # n_date - N + 1,N
    max_overnight_ret = np.nanmax(overnight_ret_array,axis = -1)# n_date - N + 1
    min_overnight_ret = np.nanmin(overnight_ret_array,axis = -1)# n_date - N + 1
    overnight_smart_array = (overnight_ret_array - min_overnight_ret[:,np.newaxis])/(max_overnight_ret[:,np.newaxis] - min_overnight_ret[:,np.newaxis])/overnight_turnover_array
    overnight_smart_array[np.isinf(overnight_smart_array)] = np.nan
    result = np.nanmean(overnight_smart_array,axis = -1)
    return result
    
class CTR_factor(object):
    def __init__(self,date_list):
        self.stock_list = ff.filter0.index
        self.date_list = date_list
        self.overnight_turnover,self.inday_turnover,self.overnight_ret,self.inday_ret = get_initial_data(self.date_list,self.stock_list)
        self.overnightsmart20 = self.overnight_smart_money()
        
    def overnight_smart_money(self):
        overnight_turnover_rolling = ff.rolling_window(self.overnight_turnover,20) # n_stock,n_date - N + 1,N
        overnight_ret_rolling = ff.rolling_window(self.overnight_ret,20) # n_stock,n_date - N + 1,N
        _num,_,_ = overnight_turnover_rolling.shape
        para_lst = [(overnight_turnover_rolling[i,:,:],overnight_ret_rolling[i,:,:]) for i in range(_num)]
        with Pool(24) as p:
            res_lst = list(tqdm(p.imap(overnight_smart_money_sub,para_lst),total = _num))# n_stock,n_date - N + 1
        overnightsmart20 = pd.DataFrame(res_lst,columns = self.date_list[19:],index = self.stock_list)
        return overnightsmart20

    def get_all_factor(self):
        # # 使用日内换手率对日内收益率进行切割（5组），求均值，计算高换手因子
        # inday_turnover_cut_inday_ret = cutting(self.inday_ret,self.inday_turnover,'max')
        # # 使用昨日换手率对隔夜收益率进行切割
        # ye_inday_turnover_cut_overnight_ret = cutting(self.inday_ret.shift(1,axis = 1),self.overnight_ret,'max')
        # # 日内收益率对日内换手率做切割
        # inday_ret_cut_inday_turnover = cutting(self.inday_turnover,self.inday_ret,'max')
        # # 次日隔夜收益率对日内换手率做切割
        # to_overnight_ret_cut_inday_turnover = cutting(self.inday_turnover,self.overnight_ret.shift(-1,axis = 1),'min')
        # # 次日隔夜换手率对日内换手率做切割
        # to_overnight_turnover_cut_inday_turnover = cutting(self.inday_turnover,self.overnight_turnover.shift(-1,axis = 1),'max')
        # 次日隔夜聪明钱对日内换手率切割，计算低隔夜聪明钱因子；做市值中性化
        CTR = cutting(self.inday_turnover.iloc[:,19:],self.overnightsmart20.shift(1,axis = 1),'min')
        # 次日隔夜聪明钱对日内换手率切割，计算低隔夜聪明钱因子；做市值中性化
        jumpCTR = cutting(self.inday_turnover.iloc[:,19:],self.overnightsmart20.shift(1,axis = 1),'jump')

        return -self.overnightsmart20,-CTR,-jumpCTR
        
        

def main():
    date_lst = ff.filter0.columns
    example1 = CTR_factor(date_lst)
    overnightsmart20,CTR,jumpCTR = example1.get_all_factor()
    mv = ff.read('total_mv').loc[:,'20200101':]
    CTR_match = CTR.reindex(index= mv.index,columns = mv.columns)
    CTR_neu = extend.spread_reg(CTR_match, mv, ind=False) # ind=True为同时进行市值与行业中性化
    jumpCTR_match = jumpCTR.reindex(index= mv.index,columns = mv.columns)
    jumpCTR_neu = extend.spread_reg(jumpCTR_match, mv, ind=False) # ind=True为同时进行市值与行业中性化
    ff.save('overnightsmart20_RC',overnightsmart20.shift(1,axis=1)*ff.filter0)
    ff.save('CTR_RC',CTR_neu.shift(1,axis=1)*ff.filter0)
    ff.save('jumpCTR_RC',jumpCTR_neu.shift(1,axis=1)*ff.filter0)
    # ff.save('overnightsmart20_RC',overnightsmart20.shift(1,axis=1)*ff.filter0)
    # ff.save('CTR_RC',CTR.shift(1,axis=1)*ff.filter0)
    # ff.save('jumpCTR_RC',jumpCTR.shift(1,axis=1)*ff.filter0)
    
if __name__ == '__main__':
    main()