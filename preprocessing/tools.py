# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Ching-Ting Kurt Lin
"""

import numpy as np

def categorise_prediction(max_ids, category):
    output = np.zeros(max_ids.shape)
    output[(max_ids == category)] = 1
    return output