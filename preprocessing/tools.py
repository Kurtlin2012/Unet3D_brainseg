# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:17:46 2020

@author: Ching-Ting Kurt Lin
"""


def categorise_prediction(max_ids, category):
    
    """
    This function is to prevent if the voxel was classified into multiple channels. 
    """
    
    import numpy as np
    
    output = np.zeros(max_ids.shape)
    output[(max_ids == category)] = 1
    return output
