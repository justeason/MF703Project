#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:19:41 2021

@author: eason
Warning: Here we have the QUARTERLY stock data, NOT DAILY!!!
"""
import pandas as pd
import numpy as np
import datetime

#df = pd.read_excel('/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/AAPLDATA2.xlsx')

"""
#determine if quarterly change in share outstanding is significant in percentage term
dropSH_OUT = df['BS_SH_OUT'].dropna()
test1 = PercentGrowth(dropSH_OUT)

#if there is no significant change in share outstanding, we fill the NaN as last shown value
newSH_OUT = df['BS_SH_OUT'].fillna(method = 'ffill')

#daily average price
pavg = (df['PX_HIGH']+df['PX_LOW'])/2
"""


#lack preferred debt, interest expense, payable, receivable, COGS, inventory, working capital


#functions to get ratios, returns dataframe
class factors:
    def __init__(self, datalocation):
        self.df = pd.read_excel(datalocation)
        self.outdf = pd.DataFrame()
        self.price = self.df['PX_LAST'].fillna(0)
        self.volumn = self.df['PX_VOLUME'].fillna(0)
        self.share = self.df['BS_SH_OUT'].dropna()
        self.bookpershare = self.df['BOOK_VAL_PER_SH'].dropna()
        self.ROE = self.df['RETURN_COM_EQY'].dropna()
        self.ROA = self.df['RETURN_ON_ASSET'].dropna()
        self.revenue = self.df['SALES_REV_TURN'].dropna()
        self.grossProfit = self.df['GROSS_PROFIT'].dropna()
        self.netDebt = self.df['NET_DEBT'].dropna()
        self.EBIT = self.df['EBIT'].dropna()
        self.EBITDA = self.df['EBITDA'].dropna()
        self.EV = self.df['CURR_ENTP_VAL'].dropna()
        self.OCF = self.df['CEST_CASHFLOW_O'].dropna()
        self.FCF = self.df['CFF_ACTIVITIES_DETAILED'].dropna()
        self.ICF = self.df['ARD_TOT_CASHFLOWS_FROM_INVESTING'].dropna()
        self.netDebt = self.df['NET_DEBT'].dropna()
        self.LTdebt = self.df['BS_LT_BORROW'].dropna()
        self.book = self.df['BOOK_VAL_PER_SH'].dropna()
        self.marketprice = self.price * self.share  #market value
        self.equity = self.share * self.bookpershare #quarterly equity
        self.netIncome = self.ROE * self.equity / 1000 #quarterly Net Income (use ROE*Equity/1000, unit in thousands)
        self.asset = self.netIncome / self.ROA * 1000 #quarterly Asset (use Net Income/ROA*1000, unit in thousands)
        
        "growth factors:"
        self.salesGrowthq = self.revenue[1:] - self.revenue.shift(1)[1:] / self.revenue.shift(1)[1:] #quarterly sales growth
        self.salesGrowth1 = self.revenue[4:] - self.revenue.shift(4)[4:] / self.revenue.shift(4)[4:] #annually sales growth
        self.salesGrowth3 = self.revenue[12:] - self.revenue.shift(12)[12:] / self.revenue.shift(12)[12:] #triannually sales growth
        self.gpGrowthq = self.grossProfit[1:] - self.grossProfit.shift(1)[1:] / self.grossProfit.shift(1)[1:] #quarterly gross profit growth
        self.gpGrowth1 = self.grossProfit[4:] - self.grossProfit.shift(4)[4:] / self.grossProfit.shift(4)[4:] #annually gross profit growth
        self.gpGrowth3 = self.grossProfit[12:] - self.grossProfit.shift(12)[12:] / self.grossProfit.shift(12)[12:] #triannually gross profit growth
        self.OCFGrowthq = self.OCF[1:] - self.OCF.shift(1)[1:] / self.OCF.shift(1)[1:] #quarterly operating cash flow growth
        self.OCFGrowth1 = self.OCF[4:] - self.OCF.shift(4)[4:] / self.OCF.shift(4)[4:] #annually operating cash flow growth
        self.OCFGrowth3 = self.OCF[12:] - self.OCF.shift(12)[12:] / self.OCF.shift(12)[12:] #triannually operating cash flow growth
        
        "profitability factors:"
        self.grossMargin = self.grossProfit / self.revenue #gross margin
        self.profitMargin = self.netIncome / self.revenue #profit margin
        self.roll4ROE = self.ROE.rolling(4,win_type='triang').sum() #annual rolling ROE
        self.roll4ROA = self.ROA.rolling(4,win_type='triang').sum() #annual rolling ROA
        #Dupond 3-factor analysis:
        self.npm = self.netIncome / self.share #net profit margin (net income/sales)
        self.at = self.revenue / self.asset    #asset turnover (sales/asset)
        self.em = self.asset/ self.equity       #equity multiplier (asset/equity)
        self.dpROE = self.npm * self.at * self.em   #dupond ROE
        self.netGearing = self.netDebt / self.asset #net gearing ratio
        self.earnpow = self.EBIT / self.asset       #basic earning power ratio
        
        "liquidit factors:"
        self.OCFratio = self.OCF / self.netDebt #operating cash flow ratio
        self.DOL = self.netIncome[1:] - (self.revenue[1:] - self.revenue.shift(1)[1:] / self.revenue.shift(1)[1:])[1:] #ddegree of operating leverage
        
        "leverage factors:"
        self.debtRatio = self.netDebt / self.asset #debt ratio
        self.LTdebtRatio = self.LTdebt / self.equity #long term debt to equity
        #market value leverage (equity + preferred + LTdebt) / market value
        #cash ratio
        #curretn ratio
        #long-term debt/working capital
        
        "efficiency factors::"
        self.assetTurn = self.netIncome / self.asset #asset turnover ratio
        #DOL degree of oeprating leverage
        #DSO days sales outstanding
        
        "market/value factors:"
        self.EPS = self.netIncome / self.share  #earning per share
        self.PE = self.marketprice / self.netIncome #Price earning ratio
        self.PCF = self.marketprice / ((self.OCF + self.FCF + self.ICF) * 1000) #price/cashflow ratio
        self.POCF = self.marketprice / (self.OCF * 1000) #price/operating cashflow ratio
        self.PB = self.marketprice / self.book #price to book ratio
        self.PS = self.marketprice / self.revenue #price/sale ratio
        self.EVS = self.EV / self.revenue #EV/sales ratio
        self.enterpriseMult = self.EV / self.EBITDA #enterprise multiple:(ev=market cap-debt+cash, ebitda=operating income-depreciation-amortization)
        
        "momentum factors:"
        #self.gain = 
        #self.loss = 
        
        "other factors:"
        self.tr = self.volumn / self.share #turnover rate (trading volumn/shares)
        #Moving Average 50 vs 200
        
    def percentGrowth(x,y):
    #input a dataframe series and a integer, returns the percent growth from previous y period
        return (x[y:] - x.shift(y)[y:]) / x.shift(y)[y:]
    """
    #quarterly Equity
    def equity(self):
        return self.share * self.bookpershare
        
    #quarterly Net Income (use ROE*Equity/1000, unit in thousands)
    def netIncome(self):
        return self.ROE * self.equity / 1000

    #quarterly Asset (use Net Income/ROA*1000, unit in thousands)
    def asset(self):
        return self.netIncome / self.ROA * 1000
    
    "growth factors:"
    def salesGrowthq(self):
        return self.percentGrowth(self.revenue, 1) #quarterly sales growth
        
    def salesGrowth1(self):
        return self.percentGrowth(self.revenue,4) #annually sales growth
        
    def salesGrowth3(self):
        return self.percentGrowth(self.revenue,12) #triannually sales growth
    
    def gpGrowthq(self):
        return self.percentGrowth(self.grossProfit,1) #quarterly gross profit growth
        
    def gpGrowth1(self):
        return self.percentGrowth(self.grossProfit,4) #annually gross profit growth
        
    def gpGrowth4(self):
        return self.percentGrowth(self.grossProfit,12) #triannually gross profit growth
    
    def OCFGrowthq(self):
        return self.percentGrowth(self.OCF,1) #quarterly OCF growth
        
    def OCFGrowth1(self):
        return self.percentGrowth(self.OCF,4) #annually OCF growth
        
    def OCFGrowth3(self):
        return self.percentGrowth(self.OCF,12) #triannually OCF growth
    
    "profitability factors:"
    def grossMargin(self):
        return self.grossProfit / self.revenue #gross margin

    def profitMargin(self):
        return self.netIncome / self.revenue #profit margin

    #annual rolling ROA/ ROE
    def roll4ROE(self):
        return self.ROE.rolling(4,win_type='triang').sum()
    
    def roll4ROA(self):
        return self.ROA.rolling(4,win_type='triang').sum()

    #Dupond 3-factor analysis:
    def npm(self):
        return self.netIncome / self.share #net profit margin (net income/sales)
    def at(self):
        return self.revenue / self.asset    #asset turnover (sales/asset)
    def em(self):
        return self.asset/ self.equity       #equity multiplier (asset/equity)
    def dpROE(self):
        return self.npm * self.at * self.em   #dupond ROE

    def netGearing(self):
        return self.netDebt / self.asset #net gearing ratio

    def earnpow(self):
        return self.EBIT / self.asset       #basic earning power ratio

    "liquidit factors:"
    def OCFratio(self):
        return self.OCF / self.netDebt #operating cash flow ratio

    def DOL(self):
        return self.netIncome[1:] / self.PercentGrowth(self.revenue,1)[1:] #ddegree of operating leverage

    "leverage factors::"
    def debtRatio(self):
        return self.netDebt / self.asset #debt ratio

    def LTdebtRatio(self):
        return self.LTdebt / self.equity #long term debt to equity

    #market value leverage (equity + preferred + LTdebt) / market value
    
    #cash ratio
    
    #curretn ratio

    #long-term debt/working capital

    "efficiency factors::"
    #DOL degree of oeprating leverage
    
    #DSO days sales outstanding
    
    def assetTurn(self):
        return self.netIncome / self.asset #asset turnover ratio

    "market/value factors:"
    def EPS(self):
        return self.netIncome / self.share  #earning per share

    def PE(self):
        return self.marketprice / self.netIncome #Price earning ratio
    
    def PCF(self):
        return self.marketprice / ((self.OCF + self.FCF + self.ICF) * 1000) #price/cashflow ratio
        
    def POCF(self):
        return self.marketprice / (self.OCF * 1000) #price/operating cashflow ratio
    
    def PB(self):
        return self.marketprice / self.book #price to book ratio

    def PS(self):
        return self.marketprice / self.revenue #price/sale ratio

    def EVS(self):
        return self.EV / self.revenue #EV/sales ratio
    
    def enterpriseMult(self):
        return self.EV / self.EBITDA #enterprise multiple:(ev=market cap-debt+cash, ebitda=operating income-depreciation-amortization)  
    
    "momentum factors:"
    #calculate the gains/losses of rolling price 
    def gainloss(self):
        return self.price.diff(1).clip(lower=0).round(2)
        self.loss = self.price.diff(1).clip(upper=0).abs().round(2)
    
    #calculate the average gains/losses of rolling price
    def avggainloss(self):
        self.avggain = self.gain.rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
        self.avgloss = self.loss.rolling(window=window_length, min_peridos=window_length).mean()[:window_length+1]
    
    
    #only for daily data, so not used here
    #price - 21/63/126/252 previous day price
    def price21(self):
        self.price21 = self.price[21:] - self.price.shift(21)[21:]
    def price63(self):
        self.price63 = self.price[63:] - self.price.shift(63)[63:]
    def price126(self):
        self.price126 = self.price[126:] - self.price.shift(126)[126:]
    def price252(self):
        self.price252 = self.price[252:] - self.price.shift(252)[252:]
    
    
    "other factors:"
    #turnover rate (trading volumn/shares)
    def tr(self):
        return self.volumn / self.share

    #Moving Average 50 vs 200
"""

df = pd.read_excel('/Users/liuyuqing/Desktop/DATA.xlsx')
price = df['PX_LAST'].fillna(0)
volume = df['PX_VOLUME'].fillna(0)
date = df['Dates'].fillna(0)

def PercentGrowth(x):
    #input a dataframe series, returns the percent growth from previous period
    return (x[1:] - x.shift()[1:])/x.shift()[1:]

#determine if quarterly change in share outstanding is significant in percentage term
dropSH_OUT = df['BS_SH_OUT'].dropna()
test1 = PercentGrowth(dropSH_OUT)

#if there is no significant change in share outstanding, we fill the NaN as last shown value
newSH_OUT = df['BS_SH_OUT'].fillna(method = 'ffill')

#daily average price
pavg = (df['PX_HIGH']+df['PX_LOW'])/2

#quarterly Equity
equity = df['BS_SH_OUT'].dropna()*df['BOOK_VAL_PER_SH'].dropna()

#quarterly Net Income (use ROE*Equity/1000, unit in thousands)
netIncome = df['RETURN_COM_EQY'].dropna()*equity/1000

#quarterly Asset (use Net Income/ROA*1000, unit in thousands)
asset = netIncome/df['RETURN_ON_ASSET'].dropna()*1000

#enterprise multiple: EV/EBITDA (ev=market cap-debt+cash, ebitda=operating income-depreciation-amortization)

#annual rolling ROA/ ROE
roll4ROE = df['RETURN_COM_EQY'].dropna().rolling(4,win_type='triang').sum()
roll4ROA = df['RETURN_ON_ASSET'].dropna().rolling(4,win_type='triang').sum()

# T21/T63/T126/T252
price21 = price[21:]-price.shift(21)[21:]
price63 = price[63:]-price.shift(63)[63:]
price126 = price[126:]-price.shift(126)[126:]
price252 = price[252:]-price.shift(252)[252:]

# Volume-price correlation factor
VPC = price.corr(volume)

# Volume impulse ratio = trading volume on the day / 
    # the average of the trading volume in the past 5 trading days
VI_r = []
for i in range(0,len(volume)):
    if i >= 5:
        mean_5d_vol = np.mean(volume[i-1]+volume[i-2]+volume[i-3]+volume[i-4]+volume[i-5])
        VI_r_i = volume[i] / mean_5d_vol
    else:
        VI_r_i = 0
    VI_r.append(VI_r_i)
    
# STD = Standard deviation of daily return 
    # (Total risk of individual stock = standard deviation of return rate in the past 30 days)
daily_return = price.ffill().pct_change().fillna(0)
STD = daily_return.rolling(30).std().fillna(0)


def macroFactors():
    #https://www.eia.gov/dnav/pet/hist/EER_EPMRU_PF4_Y35NY_DPGD.htm
    
    locations = ['/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/FEDFUNDS.csv',
                 '/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/CUSHING_OIL.csv',
                 '/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/GAS.csv',
                 '/Users/eason/OneDrive/Desktop/Boston University/Fall 2021/MF703/Final Project/CPI.xlsx'
                 ] #local files locations as a list
    
    start = '2010-01-01'  #picking data start time 
    end = '2019-12-31'
    
    results = pd.DataFrame(columns = ['interest','oil','gas','cpi'])  #put result factors series into dataframe, monthly order
    results['interest'] = interest(locations[0], start, end)
    results['oil'] = oil(locations[1], start, end)
    results['gas'] = gas(locations[2], start, end)
    results['cpi'] = cpi(locations[3], start, end)
    return results

def interest(x, start, end):
    #interest rate
    fedfund = pd.read_csv(x, index_col='DATE').sort_index() #index = date, sort by date ascending
    fedfund.index = pd.to_datetime(fedfund.index)           #convert index to datetime format as it's not in standard
    result = fedfund.loc[start:end]                         #picking rows from start date to end date
    result.index = result.index.strftime('%Y-%m')           #convert datetime index to year-month format
    return result.iloc[:,0]                                 #picking column for output

def oil(x, start, end):
    #oil price
    cushing = pd.read_csv(x, skiprows=4, index_col='Day').sort_index()
    cushing.index = pd.to_datetime(cushing.index)
    result = cushing.loc[start:end]
    result = result.resample('M').mean()   #since there are multiple data per month, we take the mean for each month
    result.index = result.index.strftime('%Y-%m')
    return result.iloc[:,0]

def gas(x, start, end):
    #gas price
    NYharbor = pd.read_csv(x, skiprows=4, index_col='Day').sort_index()
    NYharbor.index = pd.to_datetime(NYharbor.index)
    result = NYharbor.loc[start:end].resample('M').mean()
    result.index = result.index.strftime('%Y-%m')
    return result.iloc[:,0]

def cpi(x, start, end):
    #Consumer Price Index
    cpi = pd.read_excel(x, skiprows=3).T #since the year is a row, take Transpose of matrix
    cpi = cpi.iloc[2:]              #eliminate useless rows
    cpi = cpi.drop(['HALF1\n2009','HALF2\n2009','HALF1\n2010','HALF2\n2010',
              'HALF1\n2011','HALF2\n2011','HALF1\n2012','HALF2\n2012',
              'HALF1\n2013','HALF2\n2013','HALF1\n2014','HALF2\n2014',
              'HALF1\n2015','HALF2\n2015','HALF1\n2016','HALF2\n2016',
              'HALF1\n2017','HALF2\n2017','HALF1\n2018','HALF2\n2018',
              'HALF1\n2019','HALF2\n2019','HALF1\n2020','HALF2\n2020',]) #dropping the half year summary data
    cpi.index = pd.to_datetime(cpi.index)
    result = cpi.loc[start:end]
    result.index = result.index.strftime('%Y-%m')
    return result.iloc[:,0]
    
    #private equity
    
    #real estate
    
    #commodities
    
    #foreign currencies
    
    #credit
    
    #change in inflation rate
    
    #change in risk premium
    
    #fama french factors
    
    