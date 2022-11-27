#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:22:13 2021

@author: owner
"""

#中位数去极端值
#标准化序列
#单因子回归

import pandas as pd
import numpy as np
import bs4 as bs
import pandas as pd
import requests
import yfinance as yf
import datetime


from scipy.stats.mstats import winsorize
from sklearn import preprocessing
from scipy.stats import zscore
import scipy.stats as stats
from sklearn import preprocessing

import statsmodels.api as sm


"""
#yfinance get all S&p 500 stocks

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

# tickers1 = pd.DataFrame(tickers)
# tickers1.to_excel("/Users/owner/Desktop/tickers.xls", index=False)

tickers = [s.replace('\n', '') for s in tickers]
start = datetime.datetime(2019,1,1)
end = datetime.datetime(2019,7,17)
data = yf.download(tickers, start=start, end=end)

#------------------------------

test = pd.read_excel("/Users/owner/Desktop/DATA1.xlsx",header=[2,3],index_col=0)
test.index = pd.to_datetime(test.index)

#test.xs('PX_HIGH', level='Factors', axis=1)
#sma = data['Adj Close'].rolling(window=20).mean()

#prices
prices = test.xs('PX_CLOSE_1D', level='Factors', axis=1)
size = test.xs('PX_VOLUME', level='Factors', axis=1)
#prices = prices.rename(index={'Date'})

#prices log return
def log_return(prices):
    pct_change_df = prices.pct_change()
    log_return_df = np.log(1 + pct_change_df)
    return log_return_df



#因子1
sma = test.xs('PX_HIGH', level='Factors', axis=1).rolling(window=20).mean()


#数据对齐
#因子
# 因子去极值
n = 3 * 1.4826

def filter_extreme_MAD(df,n): #MAD:中位数去极值
    for column in df:
        median = df[column].quantile(0.5)
        mad = np.median(abs(df[column] - median))
        max_range = median + n * mad
        min_range = median - n * mad
        df[column] = np.where(df[column] > max_range, max_range, df[column])
        df[column] = np.where(df[column] < min_range, min_range, df[column])
    return df

#转为排序值
def rank(df):
    df = df.rank()
    return df

#因子z-score标准化
def z_score(df):
    # for column in df:
    #     df[column] = stats.zscore(df,nan_policy='omit')
    # return df
    # df.columns = [x + "_zscore" for x in df.columns.tolist()]
    df.columns = [x for x in df.columns.tolist()]
    return ((df - df.mean())/df.std(ddof=0))
    
    # cols = list(df.columns)
    # for col in cols:
    #     col_zscore = col + '_zscore'
    #     df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    # return df
    
#缺失值处理
def fill_na(df):
    # df = df.dropna(axis = 0)
    df = df.fillna(0)
    return df

#单因子回归，WLS，函数还未封装
def WLS_regression(x,y,w):
    x = sm.add_constant(x)
    model = sm.WLS(y, x_add, weights=np.sqrt(today_size)).fit()
    return model.params[]
    
#单因子回归，WLS，函数还未封装
def WLS_regression(x,y,w):
        #加权最小二乘法回归
        #w=data.iloc[:,2].tolist()
        #w=np.array([i**0.5 for i in w])
        X = sm.add_constant(x)
        regr = sm.WLS(y,X,weights=w).fit()
        #results.tvalues T值 regr.resid,残差；regr.params，beta值;results.t_test([1,0])
        return regr
        
    
returns = log_return(prices)

sma = filter_extreme_MAD(sma, n)
sma = rank(sma)
sma = z_score(sma)
sma = fill_na(sma)
print(sma)

single_factor_regression(sma)



#单因子回归
for column in sma:
    w = size[column]
    w = np.where(np.isnan(w),np.nanmean(w),w)
    w=np.array([i**0.5 for i in w])
    sma[column] = np.where(np.isnan(sma[column]),np.nanmean( sma[column]), sma[column])
    returns[column] = np.where(np.isnan(returns[column]),np.nanmean(returns[column]),returns[column])
    x_add = sm.add_constant(sma[column])
    print(returns[column])
    model = sm.WLS(returns[column], x_add, weights=w).fit()
"""    
#行业哑变量
industry = pd.read_excel("/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/Industry.xlsx",index_col=0)
industry_dummy= pd.get_dummies(industry['Sector'],drop_first=False).T



