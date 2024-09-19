#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:24:50 2024

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

# HS = [energy, metals, agriculture, others ]


class commodityAssetClass(assetClass):
    def __init__(self):
        L, MPOR
        assetClass.__init__(self, L, MPOR)
    

    
class commodityHedgingSet(hedgingSet):
    def __init__(self):
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

class commodityInstrument(instrument):
    def __init__(self, ):
        Mi, Si, Ei, Ti, size, hs, value =
        
        self.hs  = # metals, agreculture, energy, others
        instrument.__init__(self, Mi, Si, Ei, Ti, size, hs, value)
    

    



