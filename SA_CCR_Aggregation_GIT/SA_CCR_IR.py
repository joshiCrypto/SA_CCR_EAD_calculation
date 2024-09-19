#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:16:10 2024

@author: joshuakaji
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os 
import copy
from scipy.interpolate import interp1d, CubicSpline
os.chdir('/Users/joshuakaji/Desktop/freelance/Interview preparation/CCR')
# import parent classes
from SA_CCR_common import instrument, assetClass, hedgingSet
# import function 
from SA_CCR_common import position_to_str, get_delta_call

class IRInstrument(instrument):
    def __init__(self, Mi, Si, Ei, Ti, size, trd_N, ccy, value):
        self.asset_class = 'IR'
        # di = trade Notional
        instrument.__init__(self, Mi, Si, Ei, Ti, size, ccy, value)
        self.di = trd_N * get_SD(Si, Ei)
        # set maturity bucket
        # [52.57 (3)] : maturity buckets 
        if self.Mi < 1:
            self.bucket = 1
        elif self.Mi <5:
            self.bucket = 2
        else:
            self.bucket = 3

# for simplification, assume for now that all values are already converted 
def fx_convert(N, ccy):
    if ccy=='EUR':
        return N 
    if ccy=='JPY':
        return N * 0.0064
    if ccy=='USD':
        return N * 0,90
    if ccy=='GBP':
        return N * 1,18
    
# [52.34] for IR and CRE instruments => di = adj_N = trd_N * SD
def get_SD(Si, Ei):
    return (np.exp(-0.05 * Si) - np.exp(-0.05 * Ei))/0.05


########################################################################################
## IR Swap
########################################################################################
class IRSwap(IRInstrument):
    def __init__(self, T, N, size, ccy, K, freq):
        Mi, Si, Ei, Ti, trd_N = T, 0, T, np.nan, N
        value = calculate_swap_value(N, T, K, freq)
        IRInstrument.__init__(self, Mi, Si, Ei, Ti, size, trd_N, ccy, value)

    def __str__(self):
        res = "IR swap maturing in %s years" % str(self.Ei)
        return res + position_to_str(self.size)

########################################################################################
## Constructing Yield Curve - step 1) scraping data
########################################################################################
# from scipy.interpolate import interp1d, CubicSpline

# # scrap data from "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2024"
# L_df = []
# for y in range(2000, 2025):
#     url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=%i"%y
#     print(url)
#     tables = pd.read_html(url)
#     col_keep = ['Date','1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr','7 Yr', '10 Yr', '20 Yr', '30 Yr']
#     df_temp = tables[0][['Date','1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr','7 Yr', '10 Yr', '20 Yr', '30 Yr']]
#     df_temp.columns = ['date', '1M', '2M', '3M', '4M', '6M', '1Y', '2Y','3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
#     L_df.append(df_temp)

# df_all = pd.concat(L_df, axis=0)
# df_all.date = pd.to_datetime(df_all.date)
# df_all.sort_values(by='date')

# os.chdir('/Users/joshuakaji/Desktop/freelance/Interview preparation/CCR')
# df.to_pickle('treasury_spot_YC.pk')

# df = pd.read_pickle('treasury_spot_YC.pk')

########################################################################################
## Constructing Yield Curve - step 2) cubic spline
########################################################################################

fig, ax = plt.subplots(2, 1, figsize=(10, 7)) 
def convert_to_months(period):
    if period.endswith('M'):
        return int(period[:-1])
    elif period.endswith('Y'):
        return int(period[:-1]) * 12

df = pd.read_pickle('data/treasury_spot_YC.pk')

YC_date = ['2024-09-13']
YC = df.loc[df.date.isin(YC_date)].drop(columns=['date']).T
YC.columns = YC_date 
YC.plot(ax=ax[0], grid = True, color='r')

# convert to months
YC_months = copy.deepcopy(YC)
YC_months.index =  YC_months.index.map(convert_to_months)
YC_months.index.rename('months', inplace =True)

# Perform cubic spline interpolation
cubic_spline_interp = CubicSpline(YC_months.index, YC_months.iloc[:, 0])
# fill in missing months
months_fine = np.arange(1, 361)
# Interpolate values
cubic_spline_yields = cubic_spline_interp(months_fine)

YC_months_CB = pd.DataFrame({'month': months_fine, 'YC_cubic_spline': cubic_spline_yields})

# plot YC after interpolation
YC_months_CB.set_index('month').plot(grid =True, ax = ax[1])
# plot YC scatter (without interpolation)
ax[1].scatter(x=YC_months.index, y=YC_months.iloc[:, 0], marker='o', color='red')

ax[1].grid(True)
plt.show()

########################################################################################
## Swap valuation 
########################################################################################

YC = YC_months_CB.set_index('month')/100

# T in years, YC in monthly increments
def get_zcb(T): #ZCB valued at t=0
    if T==0:
        return 1
    month_position = int(T*12)
    return 1/(1 + YC.loc[month_position].values[0])**T

# t1 and t2 in years
def get_fwd_rate(t1, t2): #ZCB valued at t=0
    return (get_zcb(t1)/get_zcb(t2) - 1)/(t2 - t1)

def calculate_swap_value(N, T, K, freq = 0.5):
    dt = freq
    if T%dt != 0:
        raise ValueError("swap pricing is not defined for those parameters")
    NT = int(T//dt)
    mtm = sum([N * dt * get_zcb(n*dt) * (get_fwd_rate((n-1)*dt, n*dt) - K) for n in range(1, NT)])
    return mtm 
    
# 10Y swap with, with semi annual payoffs, K
#N=1e6
T = 10
K = 0.04
ir_swap_mtm = calculate_swap_value(1e6, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, ir_swap_mtm))
T = 10
K = 0.03
ir_swap_mtm = calculate_swap_value(1e6, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, ir_swap_mtm))


########################################################################################
## Forward Swap 
########################################################################################


class fwdSwap(IRInstrument):
    def __init__(self, fwd_start, T, N, size, ccy, K, freq):
        Mi, Si = fwd_start+T, fwd_start
        Ei, Ti, trd_N = fwd_start+T, np.nan, N
        self.T = T
        value = calculate_fwd_swap_value(N, fwd_start, T, K, freq)
        IRInstrument.__init__(self, Mi, Si, Ei, Ti, size, trd_N, ccy, value)
    def __str__(self):
        res = "%s-year interest rate swap, forward starting in %s years" % (str(self.T), str(self.Si))
        return res + position_to_str(self.size)

def calculate_fwd_swap_value(N,T_fwd_start, T_swap, K, freq = 0.5):
    dt = freq
    if T_swap%dt != 0:
        raise ValueError("fwd swap pricing is not defined for those parametors")
    NT = int(T_swap//dt)
    mtm = sum([N * dt * get_zcb(n*dt+T_fwd_start) * (get_fwd_rate((n-1)*dt+T_fwd_start, n*dt+T_fwd_start) - K) for n in range(1, NT)])
    return mtm

# 10Y swap, forward starting in 5 years with, with semi annual payoffs
T_fwd = 10
T = 10
K = 0.04
ir_fwd_swap_mtm = calculate_fwd_swap_value(1e6, T_fwd, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, forward starting in %i Years with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, T_fwd, ir_fwd_swap_mtm))
ir_swap_mtm = calculate_swap_value(1e6, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, ir_swap_mtm))


YC = YC_months_CB.set_index('month')/100

# T in years, YC in monthly increments
def get_zcb(T): #ZCB valued at t=0
    if T==0:
        return 1
    month_position = int(T*12)
    return 1/(1 + YC.loc[month_position].values[0])**T

# t1 and t2 in years
def get_fwd_rate(t1, t2): #ZCB valued at t=0
    return (get_zcb(t1)/get_zcb(t2) - 1)/(t2 - t1)

def calculate_swap_value(N, T, K, freq = 0.5):
    dt = freq
    if T%dt != 0:
        raise ValueError("swap pricing is not defined for those parametors")
    NT = int(T//dt)
    mtm = sum([N * dt * get_zcb(n*dt) * (get_fwd_rate((n-1)*dt, n*dt) - K) for n in range(1, NT)])
    return mtm 
    
# 10Y swap with, with semi annual payoffs, K
#N=1e6
T = 10
K = 0.04
ir_swap_mtm = calculate_swap_value(1e6, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, ir_swap_mtm))
T = 10
K = 0.03
ir_swap_mtm = calculate_swap_value(1e6, T, K, freq = 0.5)
print("MtM of %i Year Swap with Fixed rate %.1f%%, with semi-annual payoffs and 1MEUR Notional : %i€"%(T, K*100, ir_swap_mtm))
print(ir_swap_mtm)

########################################################################################
## Swaption (European)
########################################################################################

class euroSwaption(IRInstrument):
    def __init__(self, T_exercise, T, N, r0, K, size, ccy):
        Mi, Si = T_exercise, T_exercise
        Ei, Ti, trd_N = T_exercise+T, T_exercise, N
        self.T = T
        self.r0, self.K = r0, K
        value = calculate_euro_swaption_value()
        IRInstrument.__init__(self, Mi, Si, Ei, Ti, size, trd_N, ccy, value)

    def __str__(self):
        res = "Cash-settled European swaption referencing %s-year interest rate swap with exercise date in %s years" % (str(self.T), str(self.Si))
        return res + position_to_str(self.size)
    
    def set_delta(self): # swaptions is a call on a swap
        # supervisory volatility for IR is 50%
        sigma = 0.5
        self.delta = get_delta_call(self.r0, self.K, sigma, self.Ti) * self.size


def calculate_euro_swaption_value():
    return 0

r0 = 0.05
euro_swaption1 = euroSwaption(0.5, 5, 100e3, r0-0.01, K, 1, "USD")
euro_swaption2 = euroSwaption(0.5, 10, 100e3, r0, K, 1, "EUR")
euro_swaption3 = euroSwaption(0.5, 10, 100e3, r0+0.01, K, 1, "JPY")

print(euro_swaption1)
print(euro_swaption2)
print(euro_swaption3)

########################################################################################
## Test Aggregation with Hypothetical Portfolio
## Top down allocation of Instruments + computation of Effective Notional
## Bottom up aggregation of Effective Notional 
########################################################################################

class IR(assetClass):
    def __init__(self, L, MPOR):
        assetClass.__init__(self, L, MPOR)
        # allocate instruments into hedging sets
        self.HS = {}
        # [52.57 (2)] for IR, HS correspond to currencies
        for hs in np.unique([instr.hs for instr in self.L]):
            self.HS[hs] = IRHedgingSet([instr for instr in self.L if instr.hs==hs])
        # calculate the AddOn(asset class)
        # [52.57(7)] : IR
        self.AddOn = sum([hs.AddOnj for hs in self.HS.values()])
    
    def view_bucket_contribution(self, hs='ALL', plot=True, with_aggregate=True):
        if (type(hs) == str) & (hs !='All'):
            summary = {}
            for b in self.HS[hs].buckets.keys():
                summary[b] = self.HS[hs].buckets[b].Djk
            df = pd.DataFrame(list(summary.items()), columns=['bucket', 'effN'])
            if plot:
                df.plot(kind='bar', x='bucket', y='effN', legend=False, grid=True)
            return df
        if (type(hs) == list) | (hs=='All'):
            data = []
            if hs=='All':
                hs = list(self.HS.keys())
            for k in hs:
                for b in self.HS[k].buckets.keys():
                    data.append([k, b, self.HS[k].buckets[b].Djk])
                if with_aggregate:
                    data.append([k, 'aggregat', self.HS[k].effNj])
            df = pd.DataFrame(data, columns=['HS', 'bucket', 'effN'])
            if plot:
                df.pivot(index='HS', columns='bucket', values='effN').plot(kind='bar', grid= True)
            return df

# [52.72] supervisory factor table
class IRHedgingSet():
    def __init__(self, L): # list of instruments that ahve been defined
        hedgingSet.__init__(self, L)
        # allocate instruments into buckets
        self.buckets = {}
        for b in [1,2,3]:
            L_bucket = [inst for inst in L if inst.bucket == b]
            self.buckets[b] = IRBucket(L_bucket)
        # calculate effective notional of Hedging set
        Djx = np.matrix[self.buckets]
        # [52.57 (5)] : effective notional of the hedging set
        self.effNj = np.sqrt(self.buckets[1].Djk**2 +self.buckets[2].Djk**2 + self.buckets[3].Djk**2 \
            + 1.4*self.buckets[1].Djk*self.buckets[2].Djk + 1.4*self.buckets[2].Djk*self.buckets[3].Djk \
                +0.6*self.buckets[1].Djk*self.buckets[3].Djk)
        # calculate AddOnj of the HS (supervisory factor for IR is 0.5%)
        # [52.57 (6)] : Calculate the hedging set level add-on (SF for IR is 0.5%)
        SF = 0.005
        self.AddOnj = SF * self.effNj


class IRBucket():
    def __init__(self, L):
        # check that all intrusments have the same bucket
        if len(np.unique([inst.bucket for inst in L])) > 1:
            raise ValueError("Instantiating bucket object requires instruments to be from same bucket")
        self.L = L
        # calculate effective notional of bucket k, HS j
        # [52.57 (1)] : the effective notional Di is calculated as Di = di * MFi * δi
        # [52.57 (4)] : effective notional of a maturity bucket
        self.Djk = sum([instr.delta * instr.di * instr.MF for instr in L])
        # calculate MtM of all instruments in bucket 
        self.mtm = sum([instr.value for instr in self.L])



########################################################################################
## FX Asset Aggregation 
## FX Hedging Set Aggregation 
########################################################################################


# Interest rate or credit default swap maturing in 10 years
IRswap1 = IRSwap(10, 100e3, 1, "USD", K=0.05, freq=0.5)

IRswap2 = IRSwap(3, 100e3, -1, "EUR", K=0.05, freq=0.5)


# 10-year interest rate swap, forward starting in 5 years
fwd_swap1 = fwdSwap(5, 10, 100e3, -1, 'JPY', K=0.04, freq=0.5)
# 1-year interest rate swap, forward starting in 3 years
fwd_swap2 = fwdSwap(3, 1, 100e3, 2, 'USD', K=0.04, freq=0.5)
L = [euro_swaption1, euro_swaption2, euro_swaption3, fwd_swap1, fwd_swap2, IRswap1, IRswap2]
# margined NS with MPOR = 20d
IR1 = IR(L, 20) 
# unmargined NS
#IR2 = IR(L, np.nan)

# examine allocation of instruments to buckets 
count = 1
for ccy, hs in IR1.HS.items():
    print('\nHS : ', ccy)
    for maturity_bucket, b in hs.buckets.items():
        print('  maturity bucket ', maturity_bucket)
        for instr in b.L:
            print('      %i) ' %count, str(instr)[:50])
            #print('       Djk', b.Djk)
            count += 1
# examine the effective notional calculated for each HS (ie: ccy) of the NS
for ccy, hs in IR1.HS.items():
    print('%s effNj : '%ccy, hs.AddOnj)




df = IR1.view_HS_contribution()
df = IR1.view_instruments_contribution()

col_print = df.columns[2:]
df.description 
for i in df.instrument:
    print('#'*50)
    print(df.loc[df.instrument ==i, 'description'].values[0])
    print(df.loc[df.instrument ==i, col_print])






