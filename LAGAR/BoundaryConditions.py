#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:01:23 2019

@author: daniel
"""
import numpy as np
from LAGAR.InputFields import InputFields

class BoundaryConditions:

    def __init__(self):
        self.boundaries= []
        self.type = []

    def read_config(self,json_dict):
        self.boundaries = json_dict['boundaries']
        self.type = json_dict['type']
        if self.type == 'mask':
            self.mask_boundaries(json_dict['mask'])
        return    

    def mask_boundaries(self,json_dict):
        self.boundaries = InputFields()
        self.boundaries.get_info(json_dict)
        self.boundaries.get_mfds(json_dict)
        self.boundaries.get_get_grid(json_dict)
        self.boundaries.get_interpolants(method='nearest')

    def F(self,r):
        if self.type == 'mask':
            return self.boundaries.F_s(r)
            
        if self.type != 'mask':
            for i in range(0,len(self.boundaries)):
                if self.type[i] == 'periodic':
                    mask_0 = r[:,i] < self.boundaries[i][0] 
                    r[mask_0,i] = self.boundaries[i][1] - np.abs(r[mask_0,i] - self.boundaries[i][0])
                    mask_1 = r[:,i] > self.boundaries[i][1]
                    r[mask_1,i] = self.boundaries[i][0] + np.abs(r[mask_1,i] - self.boundaries[i][1])
                elif self.type[i] == 'fixed':
                    mask_0 = r[:,i] < self.boundaries[i][0] 
                    r[mask_0,i] = self.boundaries[i][0]
                    mask_1 = r[:,i] > self.boundaries[i][1]
                    r[mask_1,i] = self.boundaries[i][1]
                elif self.type[i] == 'void':
                    mask_0 = r[:,i] < self.boundaries[i][0] 
                    r[mask_0,i] = np.nan
                    mask_1 = r[:,i] > self.boundaries[i][1]
                    r[mask_1,i] = np.nan
                elif self.type[i] == 'none':
                    continue
        return r