#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:48:22 2024

@author: joshuakaji
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os 
import copy
import os 
os.chdir('/Users/joshuakaji/Desktop/freelance/Interview preparation/CCR')
#from SA_CCR_IR import IR
#from SA_CCR_FX import FX
#from SA_CCR_EQT import equity
# [52.31] Time period parameters: Mi, Ei, Si and Ti
class instrument():
    def __init__(self, Mi, Si, Ei, Ti, size, hs, value):
        self.Mi = Mi
        self.Si = Si
        self.Ei = Ei
        self.Ti = Ti
        self.size = size
        self.hs = hs
        self.value = value
        # di must be set by specific asset class chidren of class
    # set Maturity factor (depends on NS margin agreement)
    def set_MF(self, MPOR=np.nan): # MPOR given in days
        self.MPOR = MPOR # not used for now 
        self.margined = False if np.isnan(MPOR) else True
        # [52.48] : MF(unmargined)
        if np.isnan(MPOR): # if MPOR is NaN => Unmargined NS, Mi floored at 10 days
            # convert Mi from years to days 250
            self.MF = np.sqrt(min(max(self.Mi*250, 10), 250)/250)
        # [52.52] : MF(margined)
        # [52.50] : MPOR floored according to Centrale clearing and remargining frquency
        else: # if MPOR defined => Margined NS
        # assume Margined NS daily remargining => floor at 10 days
            self.MF = 3/2 * np.sqrt(max(MPOR, 10)/250)
    # supervisory delta assumes by default linear payoff
    def set_delta(self):
        if np.isnan(self.Ti): # if no optionality
            self.delta = self.size
        elif not ~np.isnan(self.Ti):# if Ti defined => optionality exists in intrument
            self.delta = get_delta_call(self.P, self.K, self.sigma, self.Ti) # TODO : make common in all instruments
    # must set delta and MF before setting Di
    def set_Di(self):
        try:
            self.Di = self.di * self.delta * self.MF
        except:
            raise NameError('variables di, delta or MF are not defined')

def get_delta_call(P, K, sigma, Ti):
    d1 = (np.log(P/K) + 0.5*sigma**2 * Ti)/np.sqrt(Ti)/sigma
    return norm.cdf(d1)

class assetClass():
    # when defining a HS, we must give margin agreement rules (ie : margin or no)
    def __init__(self, L, MPOR):
        self.MPOR = MPOR
        self.L = L
        # set delta and MF for all insturments
        for instr in self.L:
            instr.set_MF(self.MPOR)
            instr.set_delta()
            instr.set_Di()
        # calculate MtM of all instruments in Asset Class
        self.mtm = sum([instr.value for instr in self.L])
    # view AddOnj (ie: HS AddOn) for all HSs
    def view_HS_contribution(self, plot=True):
        summary = {}
        for k in self.HS.keys():
            summary[k] = self.HS[k].AddOnj
        df = pd.DataFrame(list(summary.items()), columns=['HS', 'AddOn(HS)'])
        if plot:
            df.plot(kind='bar', x='HS', y='AddOn(HS)', grid=True)
        return df
    def view_instruments_contribution(self, plot=True):
        data = []
        count = 1
        for instr in self.L: 
            data.append([count, str(instr), instr.di, instr.delta, instr.MF, instr.Di, instr.value])
            count+= 1
        df = pd.DataFrame(data, columns=['instrument', 'description', 'di', 'delta', 'MF', 'Di', 'mtm'])
        return df
     

class hedgingSet():
    def __init__(self, L): # list of instruments that ahve been defined
        # check that all intrusments have the same HS
        if len(np.unique([instr.hs for instr in L])) > 1:
            raise ValueError("Instantiating HS object requires instruments to be from same HS")
        self.L = L
        # calculate MtM of all instruments in Hedging Set 
        self.mtm = sum([instr.value for instr in self.L])


def position_to_str(size):
    # add direction and size info
    direction_str = 'long' if size >= 0 else 'short'
    return " // %s %i" % (direction_str, abs(size))


class nettingSet():
    def __init__(self, L, MPOR, C, mta=0, th=0):
        self.C = C # look up collateral HC prescribed by BCBS 
        # NICA represents the IM and independant amount 
        # unsegregated collateral posted by the bank is not taken into account since we assume that it will be given back in ncase of ctp default.
        self.NICA = C # look up collateral HC prescribed by BCBS 
        self.L = copy.deepcopy(L)
        self.MPOR = MPOR  
        self.L_ir = []
        self.L_fx = []
        self.L_equity = []
        
        for instr in self.L:
            if instr.asset_class == 'IR':
                self.L_ir.append(instr)
            if instr.asset_class == 'FX':
                self.L_fx.append(instr)
            if instr.asset_class == 'Equity':
                self.L_equity.append(instr)
        # create asset level aggregation nodes
        self.aggregated_nodes = {}
        self.aggregated_nodes['IR'] = IR(self.L_ir, self.MPOR)
        self.aggregated_nodes['FX'] = FX(self.L_fx, self.MPOR)
        self.aggregated_nodes['Equity'] = equity(self.L_equity, self.MPOR)
        
        # calculate AddOn at netting set level
        self.AddOnAggregated = sum([node.AddOn for node in self.aggregated_nodes.values()])
        # calculate V (MtM of the netting)
        self.V = sum([instr.value for instr in self.L if not np.isnan(instr.value) ])
        # [52.23] : calculate multiplier
        floor = 0.05
        self.m = min(1, floor + (1 - floor)*np.exp((self.V - self.C)/(2 * (1-floor)* self.AddOnAggregated)))
        # [52.20] calculate PFE
        self.PFE = self.m * self.AddOnAggregated
        # [52.3 to 52.19] : replacement cost RC
        if np.isnan(MPOR): # [52.10] for Unmargined NS
            # [52.17] : C = NICA (Net Independant Collateral Amount)
            self.RC = max(self.V - self.C, 0)
        else:
            # [52.18] : for Margined NS
            self.RC = max(self.V - self.C, th + mta - self.NICA, 0)
        # [52.1] : EAD of the netting set, alpha = 1.4 for model innacuracies (eg: wrong way risk)
        self.EAD = 1.4 * (self.RC + self.PFE)
        
    def view_asset_class_contribution(self, plot=True):
        summary = {}
        for k in self.aggregated_nodes.keys():
            summary[k] = self.aggregated_nodes[k].AddOn
        df = pd.DataFrame(list(summary.items()), columns=['asset class', 'AddOn(asset class)'])
        if plot:
            df.plot(kind='bar', x='asset class', y='AddOn(asset class)', grid=True)
        return df
    def view_instruments_contribution(self, plot=True):
        data = []
        count = 1
        for instr in self.L: 
            data.append([count, str(instr), instr.di, instr.delta, instr.MF, instr.Di, instr.value])
            count+= 1
        df = pd.DataFrame(data, columns=['instrument', 'description', 'di', 'delta', 'MF', 'Di', 'mtm'])
        return df
    def view_hedging_set_contribution(self, asset_class='ALL', plot=True, with_aggregate=True):
        if (type(asset_class) == str) & (asset_class !='All'):
            summary = {} 

            for hs in self.aggregated_nodes[asset_class].HS.keys():
                summary[hs] = self.aggregated_nodes[asset_class].HS[hs].AddOnj
            df = pd.DataFrame(list(summary.items()), columns=['Hedging Set', 'AddOn(HS)'])
            if plot:
                df.plot(kind='bar', x='bucket', y='effN', legend=False, grid=True)
            return df
        if (type(asset_class) == list) | (asset_class=='All'):
            data = []
            if asset_class=='All':
                asset_classes = list(self.aggregated_nodes.keys())
            
            for a in asset_classes:
                for hs in self.aggregated_nodes[a].HS.keys():
                    mtm = sum([instr.value for instr in self.aggregated_nodes[a].HS[hs].L])
                    data.append([a, hs, self.aggregated_nodes[a].HS[hs].AddOnj, mtm])
                if with_aggregate:
                    mtm = sum([instr.value for instr in self.aggregated_nodes[a].L])
                    data.append([a, 'aggregat', self.aggregated_nodes[a].AddOn, mtm])
            df = pd.DataFrame(data, columns=['asset class', 'hedging set', 'AddOn', 'MtM'])
            if plot:
                fig, ax = plt.subplots(nrows=2, ncols=1)
                df.pivot(index='asset class', columns='hedging set', values='AddOn').plot(kind='bar', grid= True, ax =ax[0] )
                df.pivot(index='asset class', columns='hedging set', values='MtM').plot(kind='bar', grid= True, ax =ax[1] )
            return df

    


##############################################################################
# Create Netting set of instruments 
##############################################################################

#NS0 = nettingSet(L, MPOR=np.nan, C=0, mta=0, th=0)
#NS1 = nettingSet(L, MPOR=10, C=0, mta=0, th=0)
#NS2 = nettingSet(L, MPOR=20, C=0, mta=0, th=0)
#NS3 = nettingSet(L, MPOR=40, C=0, mta=0, th=0)

#NS1 = nettingSet(L, MPOR=10, C=0, mta=0, th=0)
#NS1.view_hedging_set_contribution('All')
#df = NS1.view_hedging_set_contribution('All')
#L
#print('no Margin Agreement', NS0.PFE)
#print('MPOR = 10', NS1.PFE)
#print('MPOR = 20', NS2.PFE)
#print('MPOR = 40', NS3.PFE)


