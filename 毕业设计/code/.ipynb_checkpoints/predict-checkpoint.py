import numpy as np #导入包
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
import warnings
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from tqdm import tqdm
# 忽略特定类型的警告
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler # scaler
from tensorflow.keras.layers import LSTM,TimeDistributed,Dense,Dropout, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import tensorflow as tf
from sabr import sabr,BS
from tqdm import tqdm

date_lst = ['2023-01-01','2023-02-01','2023-03-01','2023-04-01','2023-05-01','2023-06-01','2023-07-01','2023-08-01','2023-09-01','2023-10-01','2023-11-01','2023-12-01','2024-01-01','2024-02-01','2024-03-01','2024-04-01','2024-05-01']
para_lst = ['alpha','rho','volvol','forward']
name_lst_first = ['ARIMA_first_para.xlsx','ARIMA_first_para_noweight.xlsx','new_LSTM_first_para.xlsx','new_LSTM_first_para_noweight.xlsx']
name_lst_second = ['ARIMA_second_para.xlsx','ARIMA_second_para_noweight.xlsx','new_LSTM_second_para.xlsx','new_LSTM_second_para_noweight.xlsx']     
column_lst = ['ARIMA_','ARIMA_noweight_','new_LSTM_','new_LSTM_noweight_']
def ARIMA_get_pdq(data):
    model = pm.auto_arima(data.values, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=False,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    return model.order

def get_predict_data(train_data,test_data):
    _train_data_cleaned = train_data
    _test_data_cleaned = test_data
    _p ,_d,_q= ARIMA_get_pdq(_train_data_cleaned)
    model = ARIMA(_train_data_cleaned, order=(_p,_d,_q))
    best_model = model.fit()
    forecast = best_model.predict(start=0, end=len(_train_data_cleaned) - 1)
    # 初始化预测结果列表
    predictions = []
    # 用于滚动预测的历史数据列表
    history = _train_data_cleaned.values.tolist()
    # 对测试集进行滚动预测
    for t in range(len(_test_data_cleaned)):
        # 使用历史数据进行预测
        model = ARIMA(history, order=(_p,_d,_q))
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1, typ='levels')[0]
        predictions.append(prediction)
        # 将测试集中的真实值添加到历史数据中
        history.append(_test_data_cleaned[t])
    # 将预测结果转换为Series，以便与测试集进行比较
    predictions_series = pd.Series(predictions, index=_test_data_cleaned.index)
    return predictions_series
class _ARIMA(object):
    def __init__(self,to_path):
        self.to_path = to_path
        self.pre_data = pd.read_excel(to_path,header=0,index_col=0)
    def pre_ARIMA(self):
        _pre_df = pd.DataFrame()
        for _name in para_lst:
            _pre_data_lst = []
            for i in tqdm(range(len(date_lst) - 1)):
                _train_data = self.pre_data.loc[:date_lst[i],_name]
                _test_data = self.pre_data.loc[date_lst[i]:date_lst[i+1],_name]
                _pre_data_pe = get_predict_data(_train_data,_test_data)
                _pre_data_lst.append(_pre_data_pe)
            _pre_data = pd.concat(_pre_data_lst).sort_index()
            _pre_df[_name] = _pre_data
        _save_name = 'ARIMA_'+self.to_path
        _pre_df.to_excel(_save_name)

class KerasMultiLSTM(object):

    def __init__(self,n_steps, input_size, output_size, cell_size, batch_size,drop_lr):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size # LSTM神经单元数      
        self.batch_size = batch_size # 输入batch_size大小
        self.drop_lr = drop_lr
    
    def model(self):
        
        self.model = Sequential() 
                
        # 不固定batch_size，预测时可以以1条记录进行分析
        self.model.add(LSTM(units = self.cell_size,  activation='tanh', return_sequences = True , 
                            input_shape = (self.n_steps, self.input_size))
        )
        self.model.add(Dropout(self.drop_lr))        
        # self.model.add(LSTM(units = self.cell_size, activation='tanh', return_sequences = True))
        # self.model.add(Dropout(self.drop_lr))        
        # self.model.add(LSTM(units = self.cell_size, activation='relu', return_sequences = True))
        # self.model.add(Dropout(0.1))

        #全连接，输出， add output layer
        self.model.add(TimeDistributed(Dense(self.output_size)))
        self.model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
    
    def train(self,x_train,y_train, epochs ,filename):
        history = self.model.fit(x_train, y_train, epochs = epochs, batch_size = self.batch_size).history
        self.model.save(filename)        
        return history
    
class KerasMultiLSTM_new(object):

    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, drop_lr,x0,x2,x1_a,x1_b):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.drop_lr = drop_lr
        self.x0 = x0
        self.x2 = x2
        self.x1_a = x1_a
        self.x1_b = x1_b

    def model(self):
        self.model = Sequential()
        
        # LSTM layers
        self.model.add(LSTM(units=self.cell_size, activation='tanh', return_sequences=True, input_shape=(self.n_steps, self.input_size)))
        self.model.add(Dropout(self.drop_lr))


        # Output layer with custom activation function
        self.model.add(TimeDistributed(Dense(self.output_size)))
        self.model.add(Lambda(self.custom_activation))

        self.model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

    def train(self, x_train, y_train, epochs, filename):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size).history
        self.model.save(filename)
        return history

    def custom_activation(self, x):
        x_0 = tf.where(x[:, :, 0] < tf.cast(self.x0, x.dtype), tf.cast(self.x0, x.dtype), x[:, :, 0])
        x_2 = tf.where(x[:, :, 2] < tf.cast(self.x2, x.dtype), tf.cast(self.x2, x.dtype), x[:, :, 2])
        x_1 = tf.where(x[:, :, 1] > tf.cast(self.x1_a, x.dtype), tf.cast(self.x1_a, x.dtype), x[:, :, 1])
        x_1 = tf.where(x[:, :, 1] < tf.cast(self.x1_b, x.dtype), tf.cast(self.x1_b, x.dtype), x_1)
        return tf.stack([x_0,x_1,x_2, x[:, :, 3]], axis=-1)
    
    
    
def get_batch(train_x,train_y,TIME_STEPS = 20):
    data_len = len(train_x) - TIME_STEPS+1 # may exist some problem;only suit for pred_size = 1
    seq = []
    res = []
    for i in range(data_len):
        seq.append(train_x[i:i + TIME_STEPS])
        res.append(train_y[i:i + TIME_STEPS]) #取后5组数据
        #res.append(train_y[i:i + TIME_STEPS]) 

    seq ,res = np.array(seq),np.array(res)

    return  seq, res

def get_test_train_data_new(df_data, test_len, PRED_SIZE_=1):
    sc = MinMaxScaler(feature_range= (0,1))# scale
    df_data = df_data.dropna()
    training_set = sc.fit_transform(df_data[:-test_len])
    test_set = sc.transform(df_data[-test_len:])
    h1 = sc.transform(np.array([[0.001,1,0.99,0.0001,1,1,1,1]]))
    x0 = h1[0,0]
    x1_a = h1[0,2]
    x2 = h1[0,3]
    h2 = sc.transform(np.array([[0.001,1,-0.99,0.0001,1,1,1,1]]))
    x1_b = h2[0,2]
    total_set = np.concatenate([training_set,test_set], axis=0)
    # Since it is for multiple periods of time, the data needs to be fetched separately.
    seq, res = get_batch(total_set[:-PRED_SIZE_], total_set[PRED_SIZE_:,[0,2,3,4]])
    train_seq = seq[:-test_len]
    train_res = res[:-test_len]
    test_seq = seq[-test_len:]
    test_res = res[-test_len:]
    return train_seq,train_res,test_seq,test_set[:,[1,5,6,7]],sc,df_data[-test_len:],x0,x2,x1_a,x1_b


def get_test_train_data(df_data, test_len, PRED_SIZE_=1):
    sc = MinMaxScaler(feature_range= (0,1))# scale
    training_set = sc.fit_transform(df_data[:-test_len])
    test_set = sc.transform(df_data[-test_len:])
    total_set = np.concatenate([training_set,test_set], axis=0)
    # Since it is for multiple periods of time, the data needs to be fetched separately.
    seq, res = get_batch(total_set[:-PRED_SIZE_], total_set[PRED_SIZE_:,[0,2,3,4]])
    train_seq = seq[:-test_len]
    train_res = res[:-test_len]
    test_seq = seq[-test_len:]
    test_res = res[-test_len:]
    return train_seq,train_res,test_seq,test_set[:,[1,5,6,7]],sc,df_data[-test_len:]

def get_pred_data(pred,test_y,sc):
    pred_neu = pred[:,-1,:]
    pred_neu1 = np.array([np.array([elem]) for elem in pred_neu[:,0]])
    pred_neu2 = pred_neu[:,1:]
    text_y_neu1 = np.array([np.array([elem]) for elem in test_y[:,0]])
    text_y_neu2 = test_y[:,1:]
    yy = np.concatenate((pred_neu1, text_y_neu1,pred_neu2,text_y_neu2),axis = 1)
    y=sc.inverse_transform(yy)
    return y

def pre_LSTM_new(to_path,TIME_STEPS = 20, INPUT_SIZE = 8, OUTPUT_SIZE = 4, CELL_SIZE = 24, BATCH_SIZE = 16,LR = 0.01, EPOSE = 200, PRED_SIZE = 1):
    _all_data = pd.read_excel(to_path,header=0,index_col=0).dropna()
    _save_name = 'new_LSTM_'+to_path
    pred_df_lst = []
    for i in tqdm(range(len(date_lst) - 1)):
        _para_data = _all_data.loc[:date_lst[i+1],:]
        _test_len = len(_all_data.loc[date_lst[i]:date_lst[i+1],:])
        train_x,train_y,test_x,test_y,sc,test_df,x0,x2,x1_a,x1_b = get_test_train_data_new(_para_data,_test_len,PRED_SIZE_ = PRED_SIZE)
        # 训练集需要是batch_size的倍数
        k = len(train_x)%BATCH_SIZE
        train_x,train_y = train_x[k:], train_y[k:]
        my_model = KerasMultiLSTM_new(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE,LR, x0, x2, x1_a, x1_b)
        my_model.model()
        history = my_model.train(train_x, train_y, EPOSE, "lstm-model.keras")
        pred = my_model.model.predict(test_x)
        pred_array = get_pred_data(pred,test_y,sc)
        pred_df_pe = pd.DataFrame(pred_array,columns = test_df.columns)
        pred_df_pe.index = test_df.index
        pred_df_lst.append(pred_df_pe)
    pred_df = pd.concat(pred_df_lst).sort_index()
    pred_df.to_excel(_save_name)
    
def pre_LSTM(to_path,TIME_STEPS = 20, INPUT_SIZE = 8, OUTPUT_SIZE = 4, CELL_SIZE = 24, BATCH_SIZE = 16,LR = 0.01, EPOSE = 200, PRED_SIZE = 1):
    _all_data = pd.read_excel(to_path,header=0,index_col=0).dropna()
    _save_name = 'LSTM_'+to_path
    pred_df_lst = []
    for i in tqdm(range(len(date_lst) - 1)):
        _para_data = _all_data.loc[:date_lst[i+1],:]
        _test_len = len(_all_data.loc[date_lst[i]:date_lst[i+1],:])
        train_x,train_y,test_x,test_y,sc,test_df = get_test_train_data(_para_data,_test_len,PRED_SIZE_ = PRED_SIZE)
        # 训练集需要是batch_size的倍数
        k = len(train_x)%BATCH_SIZE
        train_x,train_y = train_x[k:], train_y[k:]
        model = KerasMultiLSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE,LR)
        model.model()
        history = model.train(train_x,train_y,EPOSE,"lstm-model.keras")
        model = load_model("lstm-model.keras")
        pred = model.predict(test_x)
        pred_array = get_pred_data(pred,test_y,sc)
        pred_df_pe = pd.DataFrame(pred_array,columns = test_df.columns)
        pred_df_pe.index = test_df.index
        pred_df_lst.append(pred_df_pe)
    pred_df = pd.concat(pred_df_lst).sort_index()
    pred_df.to_excel(_save_name)    
    
def cal_downdraw(cumsum):
    downdraw=[]
    for i in range(len(cumsum)):
        lst=cumsum.iloc[:i+1]
        a=lst.max()
        b=cumsum.iloc[i]
        downdraw.append(a-b)
    return -pd.Series(downdraw).max()

def cal_returns(returns):
    result={}
    cumsum=returns.cumsum()
    result['年化收益率']=cumsum.iloc[-1]/len(cumsum)*250
    result['年化波动率']=(returns).std()*(250**0.5)
    result['夏普率']=result['年化收益率']/result['年化波动率']
    result['最大回撤']=cal_downdraw((returns).cumsum())
    result['收益回撤比']=-result['年化收益率']/result['最大回撤']
    result['胜率']=round(len(returns[returns>0])/len(returns),3)
    result['盈亏比']=-returns[returns>0].mean()/returns[returns<=0].mean()
    return result
    
def data_slice(data): # 目的，将main constract和second constract数据划分为每个期权的单独的数据
    all_strike = np.unique(data.index.get_level_values('strike').values)
    sliced_data = [] # 第一层是所有strike,第二层是所有id,数据为（real_ret,pre_ret,settle）
    id_data = []
    for _strike in all_strike:
        _strike_data = data[data.index.get_level_values('strike') == _strike]
        main_constract = np.unique(_strike_data.index.get_level_values('underlyingid').values)
        main_data = []
        id_data.append(main_constract)
        for _constract in main_constract:
            _constract_data = _strike_data[_strike_data.index.get_level_values('underlyingid') == _constract]
            _constract_trigger_event = _constract_data.trigger_event.values
            _constract_settle = _constract_data.settle.values
            _constract_position = _constract_data.position.values
            main_data.append((_constract_trigger_event,_constract_settle,_constract_position))
        sliced_data.append(main_data)
    return main_data,all_strike,id_data
        
def back_test(ret_data,ret_up,ret_down):
    # ret_data需要包括三个数据,real_ret,pre_ret,settle,position
    # 切割data为每个期权的数据
    ret_data['trigger_event'] = 0
    ret_data.loc[ret_data['pre_ret'] > ret_up, 'trigger_event'] = 1
    ret_data.loc[ret_data['pre_ret'] < ret_down, 'trigger_event'] = -1    
    ret_data['position'] = 0
    ret_data.loc[((ret_data['weight'] > 0.03) & (ret_data['weight'] <= 0.05)), 'position'] = 0.3
    ret_data.loc[((ret_data['weight'] > 0.05) & (ret_data['weight'] <= 0.1)), 'position'] = 0.6
    ret_data.loc[((ret_data['weight'] > 0.1) & (ret_data['weight'] <= 0.15)), 'position'] = 0.8
    ret_data.loc[ret_data['weight'] > 0.15, 'position'] = 1
    ret_data['strategy_ret'] = ret_data.groupby(['date','underlyingid']).transform(lambda x:np.average(x['ret'], weights=x['position']))
    ret_data['strategy_ret_all'] = ret_data.groupby(['date']).transform(lambda x:np.average(x['ret'], weights=x['position']))
    ret_all = ret_data.strategy_ret_all.groupby(level = 'date').first()
    ret_main = ret_data.strategy_ret.groupby(['date','type']).first()
    ret_second = ret_data.strategy_ret.groupby(['date','type']).last()
    main_sharpe_ratio,main_annualized_return = cal_returns(ret_main)
    second_sharpe_ratio,second_annualized_return = cal_returns(ret_second)
    all_sharpe_ratio,all_annualized_return = cal_returns(ret_all)
    return main_sharpe_ratio,main_annualized_return,second_sharpe_ratio,second_annualized_return,all_sharpe_ratio,all_annualized_return

class FORCAST(object):
    def __init__(self,name_lst_first = name_lst_first,name_lst_second = name_lst_second,column_lst = column_lst):
        self.SABR_calification = pd.read_excel('SABR_calification.xlsx',header=0,index_col=[0, 1,2,3,4])
        self.name_lst_first = name_lst_first
        self.name_lst_second = name_lst_second
        self.column_lst = column_lst
    def get_forcast_settle_ret(self):
        SABR_calification_forcast = self.SABR_calification.copy()
        for i in range(4):
            para_first_df = pd.read_excel(self.name_lst_first[i],header=0,index_col=0)
            para_second_df = pd.read_excel(self.name_lst_second[i],header=0,index_col=0)
            # 删除重复的索引
            para_first_df = para_first_df[~para_first_df.index.duplicated()]
            para_second_df = para_second_df[~para_second_df.index.duplicated()]
            for _date in tqdm(para_first_df.index):
                strike_first_lst = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 0) & (SABR_calification_forcast.index.get_level_values('date') == _date)].index.get_level_values('strike').values
                strike_second_lst = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 1) & (SABR_calification_forcast.index.get_level_values('date') == _date)].index.get_level_values('strike').values
                flag_first_lst = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 0) & (SABR_calification_forcast.index.get_level_values('date') == _date)].index.get_level_values('call_put').values
                flag_second_lst = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 1) & (SABR_calification_forcast.index.get_level_values('date') == _date)].index.get_level_values('call_put').values
                options_first_tau = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 0) & (SABR_calification_forcast.index.get_level_values('date') == _date)].options_tau.values[0]
                options_second_tau = SABR_calification_forcast[(SABR_calification_forcast.index.get_level_values('type') == 1) & (SABR_calification_forcast.index.get_level_values('date') == _date)].options_tau.values[0]
                example_first = sabr(para_first_df.loc[_date,'alpha'],0.5,para_first_df.loc[_date,'rho'],para_first_df.loc[_date,'volvol'])
                vol_first = [100*example_first.lognormal_vol(k, para_first_df.loc[_date,'forward'],options_first_tau) for k in strike_first_lst]
                settle_first_forcast = [BS(para_first_df.loc[_date,'forward'], strike_first_lst[i], options_first_tau, 0.02, vol_first[i], flag_first_lst[i]) for i in range(len(strike_first_lst))]
                example_second = sabr(para_second_df.loc[_date,'alpha'],0.5,para_second_df.loc[_date,'rho'],para_second_df.loc[_date,'volvol'])
                vol_second = [100*example_second.lognormal_vol(k, para_second_df.loc[_date,'forward'],options_second_tau) for k in strike_second_lst]
                settle_second_forcast = [BS(para_second_df.loc[_date,'forward'], strike_second_lst[i], options_second_tau, 0.02, vol_second[i], flag_second_lst[i]) for i in range(len(strike_second_lst))]
                
                SABR_calification_forcast.loc[(SABR_calification_forcast.index.get_level_values('type') == 0) & (SABR_calification_forcast.index.get_level_values('date') == _date), self.column_lst[i] + 'settle'] = settle_first_forcast
                SABR_calification_forcast.loc[(SABR_calification_forcast.index.get_level_values('type') == 1) & (SABR_calification_forcast.index.get_level_values('date') == _date), self.column_lst[i] + 'settle'] = settle_second_forcast
        SABR_calification_forcast['real_ret'] = SABR_calification_forcast['settle'].groupby(['underlyingid','call_put','strike']).transform(lambda x: x / x.shift(1) - 1)
        
        for column_name in self.column_lst:
            SABR_calification_forcast[column_name+'ret'] = SABR_calification_forcast.groupby(['underlyingid','call_put','strike']).apply(lambda x: x[column_name+'settle'] / x['settle'].shift(1) - 1).reset_index(level=[0, 1, 2], drop=True)

#         for column_name in self.column_lst:
#             SABR_calification_forcast[column_name+'ret'] = SABR_calification_forcast.groupby(['underlyingid','call_put','strike']).apply(lambda x: x[column_name+'settle'] / x['settle'].shift(1) - 1)
        SABR_calification_forcast.to_excel('SABR_calification_forcast.xlsx')    
    

        
        