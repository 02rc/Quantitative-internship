# from jgdatasdk_launcher import jgdatasdk
import sys
import pandas as pd
sys.path.append('/home/wangs/rs/lib')
import ff
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from IPython.display import display


def cal_rtn_condVaR(code):
    try:
        min_data = ff.read_min(code).close
    except:
        return np.nan
    
    if len(min_data) == 0:
        return np.nan
    else:
    
        dates = min_data.iloc[::240].index.map(lambda x:x[:10].replace('-',''))
        min_close = pd.DataFrame(min_data.values.reshape(int(len(min_data)/240),240),index=dates).T

        C = np.array([[1,-1],[0,1]])
        lnmin_close = np.log(min_close)
        miu = lnmin_close.mean()
        var = lnmin_close.std()**2
        rou = min_close.corrwith(min_close.shift(-1))

        def my_work(date):
            lnclose = np.log(min_close.loc[239,date])
            Cov = np.dot(np.dot(C,np.array([[1,rou[date]],[rou[date],1]])*var[date]),C.T)
            cond_miu = Cov[0,1]/Cov[1,1]*(lnclose-miu[date])
            cond_var = Cov[0,0] - Cov[0,1]*Cov[1,0]/Cov[1,1]
            VaR = cond_miu - 1.96*cond_var**0.5
            return VaR

        rtn_condVaR = min_close.apply(lambda x:my_work(x.name))
        return rtn_condVaR

def cal_rolling_std(df):
    return df.rolling(15,min_periods=7,axis=1).std()

def cal_rolling_std(df):
    return df.rolling(15,min_periods=7,axis=1).std()


def main():
    from multiprocessing import Pool,Manager
    from tqdm import tqdm
    
    fname = 'rtn_condVaR'
    dscr = '基于日内分钟价格数据，假设价格序列存在自相关性，构建当前与未来价格服从二元正态分布，推导得到收益率与对数股票价格的二元正态分布，在股票价格已知的情况下就可以得到收益率的条件概率分布，最终在给定置信区间内对收益率进行估计，得到置信区间的下限，并取过去15天的标准差作为最终因子值'
    src = '兴业证券'
    atr = 'jg/qza'
    
    with Pool(12) as p:
        res_lst = list(tqdm(p.imap(cal_rtn_condVaR, ff.cl), total=len(ff.cl)))
        p.close()
        p.join()
    
    rtn_condVaR = {}
    for i,item in enumerate(res_lst):
        code = ff.cl[i]
        rtn_condVaR[code] = item
        
    rtn_condVaR = cal_rolling_std(pd.DataFrame(rtn_condVaR).T)
    
    ff.save(fname,(rtn_condVaR*ff.filter0).dropna(axis=1,how='all'))
    
if __name__ == '__main__':
    main()