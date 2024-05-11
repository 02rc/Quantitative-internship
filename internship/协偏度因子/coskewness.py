import sys
sys.path.append('/home/wangs/rs/lib')
import ff
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import extend
'''
名称: CSK_XYY_UP_DOWN_120D
来源: 20240311-东北证券-因子选股系列之八：股票收益的协偏度因子
作者: RC
构造方法:
1. 构建辅助因子：
其中，上行协偏度因子的定义为:CSK_up_XYY=E[(X-E[X])(Y-E[Y])^2|Y>E[Y]] / (E[(X-E[X])^2|Y>E[Y]] * E[(Y-E[Y])^2|Y>E[Y]])
下行协偏度因子的定义为CSK_up_XYY=E[(X-E[X])(Y-E[Y])^2|Y<E[Y]] / (E[(X-E[X])^2|Y>E[Y]] * E[(Y-E[Y])^2|Y<E[Y]])
其中 X 变量代表个股收益序列, Y 变量代表市场收益序列。(这里用zz1000表示Y)
2.回溯：
CSK_XYY_UP_120D :以过去 120 个交易日为回溯窗口，计算个股收益上行协偏度。
CSK_XYY_DOWN_120D :以过去 120 个交易日为回溯窗口，计算个股收益下行协偏度。 
3.数值处理：
CSK_XYY_UP_DOWN_120D :CSK_XYY_DOWN_120D_正交化、标准化后相减。
'''
def data_rolling(_data_df,_k):
    _rolling_data = ff.rolling_window(_data_df,_k)
    if isinstance(_data_df, pd.DataFrame):
        _index_lst = _data_df.columns.values[_k-1:]
    elif isinstance(_data_df, pd.Series):
        _index_lst = _data_df.index.values[_k-1:]
    else:
        pass
    return _rolling_data,_index_lst

def get_CSK_XYY_sub(para):
    # x:(num_stock,k) , y:(k)
    _x,_y,_option = para
    _x_mean = np.nanmean(_x,axis = -1)
    _y_mean = np.nanmean(_y)
    if _option == 'all':
        _z = (_x-_x_mean[:, np.newaxis])*(((_y - _y_mean)**2))
        _y_mask_up = np.where((_y > _y_mean), 1, np.nan)
        _y_mask_down = np.where((_y < _y_mean), 1, np.nan)
        
        _x_std_up = np.nanstd(_x*_y_mask_up,axis = -1)
        _y_std_up = np.nanstd(_y*_y_mask_up)
        _CSK_XYY_up = np.nanmean(_z*_y_mask_up,axis = -1)/((_x_std_up*_y_std_up**2))
        _CSK_XYY_up[np.isinf(_CSK_XYY_up)] = np.nan
        
        _x_std_down = np.nanstd(_x*_y_mask_down,axis = -1)
        _y_std_down = np.nanstd(_y*_y_mask_down)
        _CSK_XYY_down = np.nanmean(_z*_y_mask_down,axis = -1)/((_x_std_down*_y_std_down**2))
        _CSK_XYY_down[np.isinf(_CSK_XYY_down)] = np.nan

        # 正交化
        _CSK_XYY_down_orthogonalized = _CSK_XYY_down - np.nansum(_CSK_XYY_up * _CSK_XYY_down)/np.nansum(_CSK_XYY_up * _CSK_XYY_up) * _CSK_XYY_up
        # 标准化
        _CSK_XYY_up_standardized = (_CSK_XYY_up - np.nanmean(_CSK_XYY_up)) / np.nanstd(_CSK_XYY_up)
        _CSK_XYY_down_standardized = (_CSK_XYY_down_orthogonalized - np.nanmean(_CSK_XYY_down_orthogonalized)) / np.nanstd(_CSK_XYY_down_orthogonalized)
        # 做差
        _CSK_XYY_up_down = _CSK_XYY_down_standardized - _CSK_XYY_up_standardized
        return _CSK_XYY_up_down
    else:
        if _option == 'up':
            _y_mask = np.where((_y > _y_mean), 1, np.nan)
        elif _option == 'down':
            _y_mask = np.where((_y < _y_mean), 1, np.nan)
        else:
            _y_mask = np.full_like(_y, True)
        _x_std = np.nanstd(_x*_y_mask,axis = -1)
        _y_std = np.nanstd(_y*_y_mask)
        _CSK_XYY = np.nanmean((_x-_x_mean[:, np.newaxis])*(((_y - _y_mean)**2))*_y_mask,axis = -1)/((_x_std*_y_std**2))
        _CSK_XYY[np.isinf(_CSK_XYY)] = np.nan
        return _CSK_XYY

def get_CSK_XYY(x_df,y_se,k,option):
    if np.array_equal(x_df.columns, y_se.index):
        pass
    else:
        _commom_index = (set(x_df.columns) & set(y_se.index))
        x_df = x_df.loc[:,_commom_index].sort_index(axis=1)
        y_se = y_se.loc[_commom_index].sort_index(axis=0)
    x_df_rolling,index_lst = data_rolling(x_df,k)
    _,date_num,_ = x_df_rolling.shape
    y_se_rolling,_ = data_rolling(y_se,k)
    para_lst = [(x_df_rolling[:,i,:],y_se_rolling[i,:],option) for i in range(date_num)]
    with Pool(24) as p:
        res_lst = list(tqdm(p.imap(get_CSK_XYY_sub,para_lst),total = len(para_lst)))
    # res_lst_reshape = res_lst.reshape(-1,stock_num)
    CSK_XYY_df = pd.DataFrame(res_lst,columns = x_df.index,index = index_lst).T
    return CSK_XYY_df
    
def main():
    start, end = '20170104', '20230418'
    rets_all = ff.rets_all.loc[:,start:end]
    zz1000 = pd.read_pickle('/mydata2/wangs/data/feature/zz1000.pk').loc[start:end,:]
    CSK_XYY_UP_DOWN_120D = get_CSK_XYY(rets_all,zz1000['return'],120,'all')

    mv = ff.read('total_mv').loc[:,CSK_XYY_UP_DOWN_120D.columns]
    total_index = set(CSK_XYY_UP_DOWN_120D.index) & set(mv.index)
    CSK_XYY_UP_DOWN_120D_neu = extend.spread_reg(CSK_XYY_UP_DOWN_120D.loc[total_index,:], mv.loc[total_index,:], ind=True) # ind=True为同时进行市值与行业中性化
    ff.save('CSK_XYY_UP_DOWN_120D_RC',CSK_XYY_UP_DOWN_120D_neu.shift(1,axis = 1) * ff.filter0)
if __name__ == '__main__':
    main()