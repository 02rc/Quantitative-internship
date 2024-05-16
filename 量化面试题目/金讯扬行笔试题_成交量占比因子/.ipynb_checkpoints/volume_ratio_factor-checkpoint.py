import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool

name_lst = ['交易所代码','自然日','时间','成交编号','成交代码','委托代码','BS标志','成交价格','成交数量','叫卖序号','叫买序号','空列']
def get_second(_num):
    _hour = _num//1e4
    _left_hour = _num - _hour*1e4
    _min = _left_hour//1e2
    _second = _left_hour - _min*1e2
    result = _second + _min*60 + _hour*60*60
    return result
  
def get_transaction_time_sub(_transaction_time_data):
    # 这里需要减小计算量
    if len(_transaction_time_data) == 1:
        result = 0
    else:
        _first_time = np.min(_transaction_time_data)
        _last_time = np.max(_transaction_time_data)
        _first_time_dropmillisecond = _first_time//1e3
        _last_time_dropmillisecond = _last_time//1e3
        if _first_time_dropmillisecond == _last_time_dropmillisecond:
            result = 0
        elif (_first_time_dropmillisecond >= 130000 and _last_time_dropmillisecond >= 130000) or (_first_time_dropmillisecond <= 113000 and _last_time_dropmillisecond <= 113000):
            result = get_second(_last_time_dropmillisecond) - get_second(_first_time_dropmillisecond)
        else:
            result = get_second(_last_time_dropmillisecond) - get_second(_first_time_dropmillisecond) - 5400
    return result

class long_sell_vol_prop(object):
    def __init__(self,data,name_lst = name_lst):
        self.id_BS = name_lst.index('BS标志')
        self.id_time = name_lst.index('时间')   
        self.id_sell = name_lst.index('叫卖序号')  
        self.id_vol = name_lst.index('成交数量')
        self.all_data = self.clean_data_BS_time(data)
        
    def clean_data_BS_time(self,_data):
        # 1、去掉BS标志列为nan的行；
        _nan_index = ~np.isnan(_data[:, self.id_BS]) # nan的布尔索引
        # 2、用时间列去掉九点三十分之前的所有数据；
        _sat_time_index = (_data[:,self.id_time] >= 93000000)
        _remain_index =  _nan_index*_sat_time_index
        _data_cleaned = _data[_remain_index,:]
        return _data_cleaned
    
    def get_transaction_time(self):
        # 3、统计当天每个卖单成交的起始和结束时间，计算出成交的时间间隔，精确到秒；
        # 利用np.split会快很多
        _data_sorted = self.all_data[np.argsort(self.all_data[:, self.id_sell]),:]
        _sell_index, _sell_counts = np.unique(_data_sorted[:,self.id_sell], return_index=True)
        _para_lst = np.split(_data_sorted[:, self.id_time], _sell_counts[1:])
        _res_lst = []
        for _transaction_time_data in _para_lst:
            _res_lst.append(get_transaction_time_sub(_transaction_time_data))
        # 这里不太需要并行
        # with Pool(24) as p:
        #     _res_lst = list(tqdm(p.imap(get_transaction_time_sub,_para_lst),total = len(_sell_index)))
        _sales_order_transaction_time = np.array(_res_lst)
        # 4. 将所有卖单的成交时长排序并统计出0.9分位数的值，将大于0.9分位数的卖单标记为长卖单；
        # i:从小到大排序；同时对sell_index排序
        _sorted_indices = np.argsort(_sales_order_transaction_time)
        _sales_order_transaction_time_sorted = _sales_order_transaction_time[_sorted_indices]
        _sell_index_sorted = _sell_index[_sorted_indices]
        # ii:统计出0.9分位数的值
        # 直接使用 np.percentile(sales_order_transaction_time, 90)
        _quantile_90 = np.percentile(_sales_order_transaction_time_sorted, 90)
        # iii:将大于0.9分位数的卖单标记为长卖单
        # 实际计算出来卖单的0.9分位数为0；所以就设置大于0的卖单为长买单，长卖单标记为True
        _sell_index_sorted_marked_long = (_sales_order_transaction_time_sorted > _quantile_90)
        # 输出长卖单序号
        return _sell_index_sorted[_sell_index_sorted_marked_long]
    
    def get_factor(self):
        # 长卖单序号
        _long_sell_index = self.get_transaction_time()
        # 长卖单的当天总成交量
        _long_sell_all_vol = 0
        for _long_sell_name in _long_sell_index:
            _long_sell_name_index = (self.all_data[:,self.id_sell] == _long_sell_name)
            _long_sell_name_vol = np.sum(self.all_data[_long_sell_name_index,self.id_vol])
            _long_sell_all_vol += _long_sell_name_vol
        # 当天总成交量
        _all_vol = np.sum(self.all_data[:,self.id_vol])
        _long_sell_vol_prop_factor = _long_sell_all_vol/_all_vol
        return _long_sell_vol_prop_factor
    
def final_result(_stock_data):
    example1 = long_sell_vol_prop(_stock_data)
    result = example1.get_factor()
    return result

def main():
    all_data = []
    with np.load(r"C:\Users\ASUS\rc_work\20230103.npz") as npz_file:
        stock_list = npz_file.files
        for stk in stock_list:
            all_data.append(npz_file[stk])
    with Pool(24) as p:
        factor_lst = list(tqdm(p.imap(final_result,all_data),total = len(stock_list)))
    result = np.array([stock_list,factor_lst]).T
    filename = "volume_ratio_factor.csv"
    np.savetxt(filename, result, delimiter=',',fmt='%s') 
    print(f"数据已保存到 {filename}")
if __name__ == '__main__':
    main()

  
