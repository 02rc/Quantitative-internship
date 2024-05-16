import fastbox as fb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import seaborn as sns 
from scipy.optimize import minimize
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler # scaler
from tensorflow.keras.layers import LSTM,TimeDistributed,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import warnings
from ruamel.yaml.error import UnsafeLoaderWarning

from scipy.optimize import dual_annealing

# 忽略UnsafeLoaderWarning
warnings.simplefilter('ignore', UnsafeLoaderWarning)

from multiprocessing.pool import Pool
from multiprocessing import Lock

# initialization function
def get_main_second_constract(name,from_date,to_date):
    # 1、forward 2、instrument_id 3、date 4、tau(exercise time)
    _df_maturity  = fb.data.futures.daily(underlyingid =name, 
                                          from_date=from_date,
                                          to_date=to_date, 
                                          mode="maturity"
                                         )
    _df_maturity = _df_maturity[_df_maturity['tau']>=31]
    
    _df_maturity = _df_maturity.set_index(['tau','instrumentid','settle','vol'], append=True)
    _first_second_all_data = _df_maturity['pctvol'].groupby(level = 'date').nlargest(2).index.values
    _first_all_data = _first_second_all_data[::2]
    _second_all_data = _first_second_all_data[1::2]
    _date,_,_,_first_tau,_first_id,_first_settle,_first_vol = zip(*_first_all_data)
    _,_,_,_second_tau,_second_id,_second_settle,_second_vol = zip(*_second_all_data)
    _date = np.array(_date)
    _first_tau = np.array(_first_tau)
    _first_id = np.array(_first_id)
    _first_settle = np.array(_first_settle)
    _second_tau = np.array(_second_tau)
    _second_id = np.array(_second_id)
    _second_settle = np.array(_second_settle)
    return _date,_first_tau,_first_id,_first_settle,_first_vol,_second_tau,_second_id,_second_settle,_second_vol
# Helper function
def sub_get_options_data(para):
    # This step was too slow, and I had to write a parallel
    _first_id ,_second_id ,_date,_first_settle,_second_settle  = para
    _option_df = fb.data.futuresoptions.daily(instrumentid = [_first_id,_second_id],
                                 from_date=_date.strftime('%Y-%m-%d'),
                                 to_date=_date.strftime('%Y-%m-%d')
                                )
    _option_df['options_tau'] = (_option_df.index.get_level_values('maturity') - _date).days 
    _option_df['type'] = 0 
    _option_df['forward'] = _first_settle 
    _option_df['type'][_option_df.index.get_level_values('underlyingid') == _second_id] = 1
    _option_df['forward'][_option_df.index.get_level_values('underlyingid') == _second_id] = _second_settle
    _option_df['weight'] = _option_df['vol']/_option_df['vol'].groupby(['underlyingid', 'call_put']).transform('sum')
    _option_df['option_vol'] = _option_df['vol'].groupby(level = 'underlyingid').transform('sum')
    _option_df = _option_df.set_index('type', append=True)
    _option_df = _option_df.droplevel(level=['maturity'])
    _option_df = _option_df[['settle','weight','options_tau','forward','option_vol']]
    return _option_df
# initialization function
def get_options_data(date_lst,first_id_lst,second_id_lst,first_settle,second_settle):
    # i will read all option data (May take up too much memory)
    # 1、flag 2、strike 3、date 4、futures_tau 5、settle 6、main_constract :0 ,second_constract :1
    _options_lst = []
    _para_lst = [(first_id_lst[i],second_id_lst[i],date_lst[i],first_settle[i],second_settle[i]) for i in range(len(date_lst))]
    with Pool(16) as p:
        _res_lst = list(tqdm(p.imap(sub_get_options_data,_para_lst),total = len(_para_lst)))
    _options_all = pd.concat(_res_lst,axis = 0)
    return _options_all

from scipy.stats import norm
from scipy.optimize import newton
N = norm.cdf

def BS(forward, strike, mat, r, sigma, flag):
    sign = 1
    tau = mat / 360
    discount = np.exp(- r * tau)
    if flag == "P":
        sign = -1
    std_dev = sigma * np.sqrt(tau)

    d1 = np.log(forward/strike) / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    result = discount * sign * (forward * N(sign * d1) - strike * N(sign * d2))
    return result

def IV(para):
    forward, strike, mat, r, price, flag = para
    tau = mat / 360
    discount = np.exp(- r * tau)
    def root_func(x):
        return BS(forward, strike, mat, r, x, flag) - price
    try:
        sigma = newton(root_func, 0.2)
    except:
        sigma = np.nan
    return sigma

class sabr(object):
    def __init__(self, alpha, beta, rho, volvol):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.option = True
        if (rho <= -0.991).any() or (rho >= 0.991).any():
            self.option = False
        
    def lognormal_vol(self, k, f, t):
        alpha, beta, rho, volvol = self.alpha ,self.beta ,self.rho ,self.volvol
        # Negative strikes or forwards
        if (k <= 0).any() or (f <= 0).any():
            return 0.
        eps = 1e-07
        logfk = np.log(f / k)
        fkbeta = (f*k)**(1 - beta)
        a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
        b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
        c = (2 - 3*rho**2) * volvol**2 / 24
        d = fkbeta**0.5
        v = (1 - beta)**2 * logfk**2 / 24
        w = (1 - beta)**4 * logfk**4 / 1920
        z = volvol * fkbeta**0.5 * logfk / alpha
        if (abs(z) > eps).any():
            vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * self._x(z))
            return vz
        # ATM
        else:
            v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
            return v0

    def _x(self, z):
        rho = self.rho
        a = (1 - 2*rho*z + z**2)**.5 + z - rho
        b = 1 - rho
        return np.log(a / b)

def sub_sabr_calification(para):
    _bounds = [(0.0001, 10), (-0.99, 0.99), (0.0001, 1)]
    beta,strikes,forward,mat,vols_BS = para
    def vol_square_error(x):
        sabr0 = sabr(x[0], beta, x[1], x[2])
        vols = np.array([sabr0.lognormal_vol(k_, forward, mat / 360)*100 for k_ in strikes])
        return sum((vols - vols_BS)**2)
    retries = 0
    while retries < 8:
        res = dual_annealing(vol_square_error, _bounds, maxiter=2000)
        if res.fun <= 0.01:
            alpha, rho, volvol = res.x
            sabr0 = sabr(alpha, beta, rho, volvol)
            vol_cali = np.array([sabr0.lognormal_vol(k_, forward, mat / 360)*100 for k_ in strikes])
            break
        else:
            alpha, rho, volvol = np.nan
            vol_cali = np.array([np.nan]*len(strikes))
            retries += 1
    return alpha, beta, rho, volvol, vol_cali

def sub_sabr_calification_weighted(para):
    _bounds = [(0.0001, 10), (-0.99, 0.99), (0.0001, 1)]
    beta,strikes,forward,mat,vols_BS,weight = para
    def vol_square_error(x):
        sabr0 = sabr(x[0], beta, x[1], x[2])
        vols = np.array([sabr0.lognormal_vol(k_, forward, mat / 360)*100 for k_ in strikes])
        return sum(((vols - vols_BS)*weight)**2)
    retries = 0
    while retries < 8:
        res = dual_annealing(vol_square_error, _bounds, maxiter=2000)
        if res.fun <= 0.0001:
            alpha, rho, volvol = res.x
            sabr0 = sabr(alpha, beta, rho, volvol)
            vol_cali = np.array([sabr0.lognormal_vol(k_, forward, mat / 360)*100 for k_ in strikes])
            break
        else:
            alpha, rho, volvol = np.nan
            vol_cali = np.array([np.nan]*len(strikes))
            retries += 1
    return alpha, beta, rho, volvol, vol_cali
    
class SABR(object):
    def __init__(self,name,from_date = '2020-01-01',to_date = '2024-04-26'):
        # Verify the validity of variables
        self.name = name
        try:
            self.date_lst,self.first_tau,self.first_id,_first_settle,self.first_future_vol,self.second_tau,self.second_id,_second_settle,self.second_future_vol = get_main_second_constract(name,from_date,to_date)
            self.option_all = get_options_data(self.date_lst,self.first_id,self.second_id,_first_settle,_second_settle)
        except Exception as e:
            print('Data initialization error:',e)
        
    def black_scholes(self,r=0.02):
        _price_all = self.option_all.settle.values
        _mat_all = self.option_all.options_tau.values
        _flag_all = self.option_all.index.get_level_values('call_put').values
        _forward_all = self.option_all.forward.values
        _strike_all = self.option_all.index.get_level_values('strike').values
        _para_lst = [(_forward_all[i], _strike_all[i], _mat_all[i], r, _price_all[i], _flag_all[i]) for i in range(len(_price_all))]
        with Pool(16) as p:
            _IV_lst = list(tqdm(p.imap(IV,_para_lst),total = len(_para_lst)))
        self.option_all['volatility_BS'] = _IV_lst
        self.option_all.to_excel('black_scholes_df.xlsx')
        
    def sabr_calification(self,if_weight = True,beta = 0.5):
        if 'volatility_BS' in self.option_all.columns:
            pass
        else:
            self.black_scholes()
        _option_all_cleaned = self.option_all[self.option_all['weight'] > 0.03].dropna(how = 'any',axis = 0)
        if if_weight:
            # Estimated intervals for alpha, rho, and volvol
            _para_list = [(beta,group.index.get_level_values('strike').values,group['forward'].iloc[0],group['options_tau'].iloc[0],group['volatility_BS'].values,group['weight'].values) for name, group in _option_all_cleaned.groupby(['date', 'underlyingid'])]
            with Pool(16) as p:
                _res_lst = list(tqdm(p.imap(sub_sabr_calification_weighted,_para_list),total = len(_para_list)))
            _alpha_lst, _beta_lst, _rho_lst, _volvol_lst, _vol_cali_lst = zip(*_res_lst)
            _sabr_first_para_df = pd.DataFrame([_alpha_lst[::2],_beta_lst[::2],_rho_lst[::2],_volvol_lst[::2]],columns = self.date_lst,index = ['alpha','beta','rho','volvol']).T
            _sabr_first_para_df['forward'] = (_option_all_cleaned.forward.groupby(level = 'date').first()).values
            _sabr_first_para_df['options_tau'] = (_option_all_cleaned.options_tau.groupby(level = 'date').first()).values
            _sabr_first_para_df['option_vol'] = (_option_all_cleaned.option_vol.groupby(level = 'date').first()).values
            _sabr_first_para_df['future_vol'] = self.first_future_vol
            
            _sabr_second_para_df = pd.DataFrame([_alpha_lst[1::2],_beta_lst[1::2],_rho_lst[1::2],_volvol_lst[1::2]],columns = self.date_lst,index = ['alpha','beta','rho','volvol']).T
            _sabr_second_para_df['forward'] = (_option_all_cleaned.forward.groupby(level = 'date').last()).values
            _sabr_second_para_df['options_tau'] = (_option_all_cleaned.options_tau.groupby(level = 'date').last()).values
            _sabr_second_para_df['option_vol'] = (_option_all_cleaned.option_vol.groupby(level = 'date').last()).values
            _sabr_second_para_df['future_vol'] = self.second_future_vol
            
            _vol_cali = np.concatenate(_vol_cali_lst)
            _option_all_cleaned['volatility_Cali_weighted'] = _vol_cali
            self.first_sabr_para = _sabr_first_para_df
            self.second_sabr_para = _sabr_second_para_df
            if hasattr(self, 'option_all_cali'):
                self.option_all_cali['volatility_Cali_weighted'] = _vol_cali
            else:
                self.option_all_cali = _option_all_cleaned
            _option_all_cleaned.to_excel('SABR_calification.xlsx')
            _sabr_first_para_df.to_excel('first_para.xlsx')
            _sabr_second_para_df.to_excel('second_para.xlsx')
            
        else:
            _para_list = [(beta,group.index.get_level_values('strike').values,group['forward'].iloc[0],group['options_tau'].iloc[0],group['volatility_BS'].values) for name, group in _option_all_cleaned.groupby(['date', 'underlyingid'])]
            with Pool(16) as p:
                _res_lst = list(tqdm(p.imap(sub_sabr_calification,_para_list),total = len(_para_list)))
            _alpha_lst, _beta_lst, _rho_lst, _volvol_lst, _vol_cali_lst = zip(*_res_lst)
            
            _sabr_first_para_df_no = pd.DataFrame([_alpha_lst[::2],_beta_lst[::2],_rho_lst[::2],_volvol_lst[::2]],columns = self.date_lst,index = ['alpha','beta','rho','volvol']).T
            _sabr_first_para_df_no['forward'] = (_option_all_cleaned.forward.groupby(level = 'date').first()).values
            _sabr_first_para_df_no['options_tau'] = (_option_all_cleaned.options_tau.groupby(level = 'date').first()).values
            _sabr_first_para_df_no['option_vol'] = (_option_all_cleaned.option_vol.groupby(level = 'date').first()).values
            _sabr_first_para_df_no['future_vol'] = self.first_future_vol
            
            _sabr_second_para_df_no = pd.DataFrame([_alpha_lst[1::2],_beta_lst[1::2],_rho_lst[1::2],_volvol_lst[1::2]],columns = self.date_lst,index = ['alpha','beta','rho','volvol']).T
            _sabr_second_para_df_no['forward'] = (_option_all_cleaned.forward.groupby(level = 'date').last()).values
            _sabr_second_para_df_no['options_tau'] = (_option_all_cleaned.options_tau.groupby(level = 'date').last()).values
            _sabr_second_para_df_no['option_vol'] = (_option_all_cleaned.option_vol.groupby(level = 'date').last()).values
            _sabr_second_para_df_no['future_vol'] = self.second_future_vol
            _vol_cali = np.concatenate(_vol_cali_lst)
            _option_all_cleaned['volatility_Cali'] = _vol_cali
            if hasattr(self, 'option_all_cali'):
                self.option_all_cali['volatility_Cali'] = _vol_cali
            else:
                self.option_all_cali = _option_all_cleaned
            _option_all_cleaned.to_excel('SABR_calification_noweight.xlsx')
            _sabr_first_para_df_no.to_excel('first_para_noweight.xlsx')
            _sabr_second_para_df_no.to_excel('second_para_noweight.xlsx')                
                
            
            
            
            
            
            
         
                       
                       


        