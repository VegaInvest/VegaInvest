import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pandas_datareader.data as web
from numpy import matlib

def Import_data_inputs(startdate, tickers):
    index = len(tickers)
    
    stocks_rets = Import_stocks(startdate, tickers)
    factor_rets = Import_factors(startdate)
    
    merged = pd.merge(left = stocks_rets, left_index=True,
                      right = factor_rets, right_index=True,
                      how='inner')

    stocks_rets = merged.iloc[:,:index]
    factor_rets = merged.iloc[:,index:]
    
    rfr = factor_rets.iloc[:,5]
    excess_rets = stocks_rets.subtract(rfr, axis = 0)
    
    return excess_rets, factor_rets

    
def Import_factors(startdate):
    url = urlopen("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

    #Download FFdata
    zipfile = ZipFile(BytesIO(url.read()))
    FFdata = pd.read_csv(zipfile.open('F-F_Research_Data_5_Factors_2x3_daily.CSV'), 
                         header = 0, names = ["Date",'MKT-RF','SMB','HML','RMW','CMA','RF'], 
                         skiprows=3)
    FFdata=FFdata.loc[FFdata["Date"]>=startdate].set_index("Date")


    #Download momentum 
    url = urlopen("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")

    #Download Zipfile and create pandas DataFrame
    zipfile = ZipFile(BytesIO(url.read()))
    Momdata = pd.read_csv(zipfile.open('F-F_Momentum_Factor_daily.CSV'),  
                         header = 0, names = ["Date",'Mom'], 
                         skiprows=13)[:-1]
    Momdata['Date']=Momdata['Date'].astype(int)
    Momdata=Momdata.loc[Momdata["Date"]>=startdate].set_index("Date")

    FFdata=FFdata.join(Momdata)/100
    FFdata.index = pd.to_datetime(FFdata.index, format='%Y%m%d')
    
    return FFdata


def Import_stocks(startdate, tickers):
    startdate = startdate-1
    startdate = str(startdate)[4:6]+'/'+str(startdate)[6:8]+'/'+str(startdate)[0:4]

    stock_ret = web.get_data_yahoo(tickers, startdate, interval='d')['Adj Close'].dropna()
    stock_ret = stock_ret/stock_ret.shift(1) - 1 # convert prices to daily returns
    stock_ret = stock_ret[1:]
    
    return stock_ret

def Import_factors(startdate):
    url = urlopen("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

    #Download FFdata
    zipfile = ZipFile(BytesIO(url.read()))
    FFdata = pd.read_csv(zipfile.open('F-F_Research_Data_5_Factors_2x3_daily.CSV'), 
                         header = 0, names = ["Date",'MKT-RF','SMB','HML','RMW','CMA','RF'], 
                         skiprows=3)
    FFdata=FFdata.loc[FFdata["Date"]>=startdate].set_index("Date")


    #Download momentum 
    url = urlopen("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")

    #Download Zipfile and create pandas DataFrame
    zipfile = ZipFile(BytesIO(url.read()))
    Momdata = pd.read_csv(zipfile.open('F-F_Momentum_Factor_daily.CSV'),  
                         header = 0, names = ["Date",'Mom'], 
                         skiprows=13)[:-1]
    Momdata['Date']=Momdata['Date'].astype(int)
    Momdata=Momdata.loc[Momdata["Date"]>=startdate].set_index("Date")

    FFdata=FFdata.join(Momdata)/100
    FFdata.index = pd.to_datetime(FFdata.index, format='%Y%m%d')
    
    return FFdata

def Param_forecast(input_stock_rets, input_factor_rets, lookback, forecast, model):
    if forecast>lookback:
        print("Warning! Increase lookback length to display full forecast.")
    forecast = min(forecast,lookback)

    input_stock_rets = np.array(input_stock_rets)
    input_factor_rets = np.array(input_factor_rets)
    
    num_assets = input_stock_rets.shape[1]
    F = factor_forecast(input_factor_rets, lookback, forecast)
    mu = []
    
    for i in range(num_assets):
        mu = mu + [beta_forecast(input_factor_rets, F, input_stock_rets.transpose()[i], lookback, forecast, model)[2]]
        
    mu = np.array(mu)
    Q = cov_forecast(input_stock_rets, mu.transpose(), lookback, forecast)   
        
    return mu, Q
    
def cov_forecast(rets_historical, rets_forecast, lookback, forecast):    
    Q = []
    rets = np.vstack([rets_historical[-lookback:], rets_forecast])

    for i in range(forecast):
        Q = Q + [np.cov(rets[i:i+lookback].transpose())]
        
    Q = np.array(Q)
    
    return Q

def factor_forecast(factor_rets, lookback, forecast):
    output = np.full((forecast,factor_rets.shape[1]), np.nan)
    rolling = factor_rets[-lookback:]

    for i in range(forecast):
        #output[i] = np.array(pd.DataFrame(rolling).mean())
        output[i] = np.array(gmean(pd.DataFrame(rolling)+1)-1)
        rolling = np.vstack([rolling[1:],output[i].transpose()])
    return output

def beta_forecast(historical_factor_rets, factor_forecast, single_stock_ret, lookback, forecast, model): #single stock forecast
    betas = np.full((forecast,factor_forecast.shape[1]), np.nan)
    alphas = np.full(forecast, np.nan)
    rolling = single_stock_ret[-lookback:]
    factors = np.vstack([historical_factor_rets[-lookback:], factor_forecast])

    for i in range(forecast):      
        x = model.fit(factors[i : i + lookback], rolling)
        betas[i]  = x.coef_
        alphas[i]  = x.intercept_ 
        ret = np.matmul(betas[i],factor_forecast[i])+alphas[i]
        rolling = np.append(rolling[1:],[ret])
        mu = rolling[lookback-forecast:]
    
    return alphas, betas, mu