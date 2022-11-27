#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 13:04:13 2021

@author: eason
"""
#https://www.mlq.ai/capital-asset-pricing-model-python/
#https://www.alpharithms.com/predicting-stock-prices-with-linear-regression-214618/
import pandas as pd
import numpy as np
import math
import json
import yfinance as yf
import pandas_ta
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

start = '2000-11-18'
end = '2021-11-19'

spy = yf.download('spy', start, end, adjusted=True)
ilmn = yf.download('ilmn', start, end, adjusted=True)
cmg = yf.download('cmg', start, end, adjusted=True)
msci = yf.download('msci', start, end, adjusted=True)
rmd = yf.download('rmd', start, end, adjusted=True)
ftnt = yf.download('ftnt', start, end, adjusted=True)
zbra = yf.download('zbra', start, end, adjusted=True)
ter = yf.download('ter', start, end, adjusted=True)
rol = yf.download('rol', start, end, adjusted=True)
abmd = yf.download('abmd', start, end, adjusted=True)
ua = yf.download('ua', start, end, adjusted=True)
a = spy['Adj Close'].resample('M').last().pct_change().dropna()
b = ilmn['Adj Close'].resample('M').last().pct_change().dropna()
c = cmg['Adj Close'].resample('M').last().pct_change().dropna()
d = msci['Adj Close'].resample('M').last().pct_change().dropna()
e = rmd['Adj Close'].resample('M').last().pct_change().dropna()
f = ftnt['Adj Close'].resample('M').last().pct_change().dropna()
g = zbra['Adj Close'].resample('M').last().pct_change().dropna()
h = ter['Adj Close'].resample('M').last().pct_change().dropna()
i = rol['Adj Close'].resample('M').last().pct_change().dropna()
j = abmd['Adj Close'].resample('M').last().pct_change().dropna()
k = ua['Adj Close'].resample('M').last().pct_change().dropna()
A = pd.DataFrame(a)
a.name = 'spy'
b.name = 'ilmn'
c.name = 'cmg'
d.name = 'msci'
e.name = 'rmd'
f.name = 'ftnt'
g.name = 'zbra'
h.name = 'ter'
i.name = 'rol'
j.name = 'abmd'
k.name = 'ua'
data = [a,b,c,d,e,f,g,h,i,j,k]
oil = pd.read_csv('BrentOilPrices.csv')

A.ta.ema(close = 'Adj Close', length=10, append=True)
A = A.iloc[10:]
X_train, X_test, y_train, y_test = ms.train_test_split(A[['Adj Close']], A[['EMA_10']], test_size=.2)
model = LinearRegression()
model.fit(X_train, y_train)
metrics = [model.coef_, mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)]
