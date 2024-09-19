#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:53:55 2024

@author: joshuakaji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import os 
# import parent classes
from SA_CCR_common import instrument, assetClass, hedgingSet
# import function 
from SA_CCR_common import position_to_str, get_delta_call


class equityInstrument(instrument):
    def __init__(self, Mi, Si, Ei, Ti, size, hs, value, trd_N):
        self.asset_class = 'Equity'
        instrument.__init__(self, Mi, Si, Ei, Ti, size, hs, value)
        self.di = trd_N

class euroCallEquity(equityInstrument):
    def __init__(self, S0, K, T, name, size, single_name, N):
        Mi = T
        Si = T
        Ei = T
        Ti = T
        hs = name
        trd_N = S0 * N
        self.K = K
        self.S0 = S0
        # TODO
        value = black_scholes_call(N, S0, sigma, K, T)
        hs = name
        # single name influences the volatility and systemic correlation factors 
        self.single_name = single_name
        equityInstrument.__init__(self, Mi, Si, Ei, Ti, size, hs, value, trd_N)
    def __str__(self):
        res = "euro call on %s with exercise date in %.1f years" %(self.hs, self.Ti)
        res += position_to_str(self.size)
        return res
    def set_delta(self):
        # [52.72] : supervisory vol for single names is 120%, 75% for index 
        sigma = 1.2 if self.single_name else 0.75
        self.delta = self.size * get_delta_call(self.S0, self.K, sigma, self.Ti)

# YC contructed in via cubic spline in SA_CCR_IR
YC_months_CB = pd.read_pickle('data/YC_ZcB_monthly_cubic_spline.pk')
# assume for simplicity that calls are always on 1 share
def black_scholes_call(N, S, sigma, K, T):
    r=YC_months_CB.loc[int(T*12)].values[0]
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0)) - (K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call_price * N

# european call option on underlying equity spot price maturing in 6 months 
S0 = 90
K = 100
T = 0.5
name = 'aapl'
N = 1000 # number of underlyings in the call
single_name = True
sigma = 0.3
euroCall1 = euroCallEquity(S0, K, T, name, 1, single_name, N)
name = 'tesla'
euroCall2 = euroCallEquity(S0, K, T, name, 2, single_name, N)

name = 'tesla'
euroCall3 = euroCallEquity(S0, K, T, name, -1, single_name, N)

name = 'snp500'
single_name = False
euroCall4 = euroCallEquity(S0, K, T, name, -2, single_name, N)

print(euroCall1)
print(euroCall2)
print(euroCall3)
print(euroCall4)

########################################################################################
## Equity Asset Aggregation 
## Equity Hedging Set Aggregation 
########################################################################################

class equity(assetClass):
    def __init__(self, L, MPOR):
        # allocate instruments into hedging sets
        self.HS = {}
        assetClass.__init__(self, L, MPOR)
        # [52.59 (2)] for equity, hedging sets correspond to entities
        for hs in np.unique([instr.hs for instr in self.L]):
            self.HS[hs] = equityHedgingSet([instr for instr in self.L if instr.hs==hs])
        # calculate the AddOn(asset class)
        # [52.56 (4)] : calculate AddOnEquity by aggregate across entities/HS 
        # [52.72] : systematic correlation param depends on whether index or single name
        rho_single_name = 0.5
        rho_index = 0.8
        sum1_single_names = [hs.AddOnj * rho_single_name for hs in self.HS.values() if hs.single_name]
        sum1_index = [hs.AddOnj * rho_index for hs in self.HS.values() if hs.single_name]
        sum1 = sum(sum1_single_names + sum1_index)**2
        sum2_single_names = sum([hs.AddOnj * (1 - rho_single_name**2) for hs in self.HS.values() if hs.single_name])
        sum2_index = sum([hs.AddOnj**2 * (1 - rho_index**2) for hs in self.HS.values() if not hs.single_name])
        sum2 = sum2_index + sum2_single_names
        self.AddOn = np.sqrt(sum1 + sum2)

class equityHedgingSet(hedgingSet):
    def __init__(self, L):
        self.single_name = L[0].single_name 
        hedgingSet.__init__(self, L)
        # [52.66 (1) & (2)] calculate effective notional of Hedging set
        self.effNj = sum([instr.di * instr.delta * instr.MF for instr in L])
        # [52.66 (3)] : Calculate the hedging set entity level add-on 
        # [52.72] : SF for Equity depends on whether underlying is index or single name 
        SF = 0.32 if self.single_name else 0.2
        self.AddOnj = self.effNj * SF

########################################################################################
## Test Aggregation with Hypothetical Portfolio
########################################################################################

L = [euroCall1, euroCall2, euroCall3, euroCall4]

EQT1 = equity(L, MPOR=10)

df = EQT1.view_HS_contribution()
df = EQT1.view_instruments_contribution()

col_print = df.columns[2:]
for i in df.instrument:
    print('#'*50)
    print(df.loc[df.instrument ==i, 'description'].values[0])
    print(df.loc[df.instrument ==i, col_print])

