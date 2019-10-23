#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:15:48 2019

@author: daniel
"""
from LAGAR.InitialConditions import InitialConditions,Emissors
from LAGAR.Solvers import Solvers
from LAGAR.BoundaryConditions import BoundaryConditions
from LAGAR.Writer import Writer
from LAGAR import Kernels
import json
import pandas as pd
import numpy as np
import copy



class LAGARCore:
    
    def __init__(self,json_file):
        self.json_file = json_file
        self.initial_conditions = []
        self.input_fields = []
        self.solver = []
        self.writer_output = []
        self.emissors = []
        self.boundaries = []
        self.kernel  = []
        self.setup = []
   
    def set_initial_conditions(self):
        self.initial_conditions = InitialConditions()
        self.initial_conditions.generate(self.json_dict['r0'])

    def set_writer(self):
        self.writer_output = Writer(self.initial_conditions)
        self.writer_output.read_config(self.json_dict['output'])
        
    def set_kernel(self):
        self.kernel = getattr(Kernels, self.json_dict['kernel']['kernel'])(self.json_dict['input_fields'])
    
    def set_solver(self):
        self.solver = Solvers(self.initial_conditions, self.kernel, self.writer_output, self.boundaries,self.emissors)
        self.solver.solver_type = self.json_dict['solver']['solver']
    
    def set_boundaries(self):
        self.boundaries = BoundaryConditions()
        self.boundaries.read_config(self.json_dict['boundaries'])

    def set_emissor(self):
        self.emissors = Emissors()
        self.emissors.read_config(self.json_dict['emissors'])
          
    def read_simulation(self):
        file_setup = open(self.json_file)  # 
        self.json_dict = json.load(file_setup)
        file_setup.close
    
    def set_simulation(self):
        self.set_initial_conditions()
        self.set_writer()
        self.set_kernel()
        if "boundaries" in self.json_dict: 
            self.set_boundaries()
            self.kernel.boundaries = self.boundaries

        if "emissors" in self.json_dict: 
            self.set_emissor()
        self.set_solver()
    
    def update_simulation(self):
        self.initial_conditions.update(self.json_dict['r0'])
        self.set_kernel()
        self.set_solver()
    
    def run_simulation(self):
        self.solver.Integrate()



    
class LAGAR_Parallel:
    
    def __init__(self,json_dict):
        self.LAGAR_father = LAGARCore(json_dict)
        self.LAGAR_sons = []
        self.number_of_sons = []
        self.time_intervals = []
        self.number_of_chunks = []
    
    
    
    def split_time_chunks(self):
        self.LAGAR_father.read_simulation()

        t_start = self.LAGAR_father.json_dict['r0']['time_range'][0]
        t_end = self.LAGAR_father.json_dict['r0']['time_range'][1]
        freq = self.LAGAR_father.json_dict['r0']['time_range'][2]
        
        timespan =  pd.date_range(start=t_start, end=t_end, freq=freq)
        self.time_intervals = np.array_split(timespan,self.number_of_chunks)

        for k in range(0,len(self.time_intervals)-1):
            self.time_intervals[k] = self.time_intervals[k].union([self.time_intervals[k][-1] + self.time_intervals[k][-1].freq])
            
   
    
    def create_father_simulation(self):
        print('************************')
        print('LAGAR father simulation')
        print('************************')
        print('Setting integration interval')
        
        t_start_int = self.time_intervals[0][0]
        t_end_int = self.time_intervals[0][-1]
        freq_int = self.LAGAR_father.json_dict['r0']['time_range'][2]
        time_range_integration = [t_start_int.strftime('%Y/%m/%d %H:%M:%S'),t_end_int.strftime('%Y/%m/%d %H:%M:%S'),freq_int]
        self.LAGAR_father.json_dict['r0']['time_range'] = time_range_integration
        
        print('t_start:',t_start_int.strftime('%Y/%m/%d %H:%M:%S'))
        print('t_end:',t_end_int.strftime('%Y/%m/%d %H:%M:%S'))
        
        print('Setting data interval')
        #We add one extra timestep to avoid the problems when the integration finished just or close when the data finished.
        t_start_dat = self.time_intervals[0][0]
        t_end_dat = self.time_intervals[0][-1] +  t_start_dat.freq
        print('t_start:',t_start_dat.strftime('%Y/%m/%d %H:%M:%S'))
        print('t_end:',t_end_dat.strftime('%Y/%m/%d %H:%M:%S'))
         
        # We set the father simulation

        self.LAGAR_father.json_dict['input_fields']['velocity']['dims_slice']['time']= [t_start_dat.strftime('%Y/%m/%d %H:%M:%S'),\
                                   t_end_dat.strftime('%Y/%m/%d %H:%M:%S'),None]
        self.LAGAR_father.set_simulation()
        self.LAGAR_father.run_simulation()
    
        return
    
    def update_father(self,time_interval):
        print('************************')
        print('LAGAR father simulation')
        print('************************')
        print('Setting integration interval')
        
        t_start_int = time_interval[0]
        t_end_int = time_interval[-1]
        freq_int = self.LAGAR_father.json_dict['r0']['time_range'][2]
        time_range_integration = [t_start_int.strftime('%Y/%m/%d %H:%M:%S'),t_end_int.strftime('%Y/%m/%d %H:%M:%S'),freq_int]
        self.LAGAR_father.json_dict['r0']['time_range'] = time_range_integration
        
        print('t_start:',t_start_int.strftime('%Y/%m/%d %H:%M:%S'))
        print('t_end:',t_end_int.strftime('%Y/%m/%d %H:%M:%S'))
        
        print('Setting data interval')
        #We add one extra timestep to avoid the problems when the integration finished just or close when the data finished.
        t_start_dat = time_interval[0]
        t_end_dat = time_interval[-1] +  t_start_dat.freq
        print('t_start:',t_start_dat.strftime('%Y/%m/%d %H:%M:%S'))
        print('t_end:',t_end_dat.strftime('%Y/%m/%d %H:%M:%S'))

         
        # We set the father simulation
        self.LAGAR_father.json_dict['input_fields']['velocity']['dims_slice']['time']= [t_start_dat.strftime('%Y/%m/%d %H:%M:%S'),\
                                   t_end_dat.strftime('%Y/%m/%d %H:%M:%S'),None]
        
        self.LAGAR_father.update_simulation()
        self.LAGAR_father.run_simulation()
        
        return


    def run_simulation(self):
        self.split_time_chunks()
        self.create_father_simulation()
        for time_interval in self.time_intervals[1:]:
            print(time_interval)
            self.update_father(time_interval)







#    def create_sons(self):
#        
#        r0_splitting = np.array_split(self.LAGAR_father.initial_conditions.r0,self.number_of_sons)
#        LAGAR_sons = []
#        print('From father simulations to sons simulations')
#        for son_id in range(0,self.number_of_sons):
#            self.LAGAR_father.initial_conditions.r0 = r0_splitting[son_id]
#            LAGAR_sons.append(copy.deepcopy(self.LAGAR_father))
#        son_id=0
#        print('Creating the sons simulations')
#        for son in LAGAR_sons:
#            #son.json_dict['output']['path'] = son.writer_output.path +str(son_id)
#            son.json_dict['output']['pattern'] = 'block_'+str(son_id).zfill(2)+'_' + son.writer_output.pattern
#            son_id = son_id +1
#            son.set_writer()
#        return LAGAR_sons
#
#
#    @staticmethod
#    def update_son(LAGAR_son,time_section):
#        t_start_str = time_section[0].strftime('%Y/%m/%d %H:%M:%S')
#        t_end_str = time_section[-1].strftime('%Y/%m/%d %H:%M:%S')
#        freq =  time_section[0].freqstr
#        time_range = [t_start_str,t_end_str,freq]
#        print(time_range)
#        print('Updating sons simulations')
#        LAGAR_son.json_dict['r0']['time_range'] = time_range
#        LAGAR_son.json_dict['input_fields']['velocity']['dims_slice']['time']=[time_range[0],time_range[1],None]
#        LAGAR_son.initial_conditions.span_time()
#        LAGAR_son.set_kernel()
#        LAGAR_son.set_solver()
#        
#        return LAGAR_son
#    
#
#    
# 
#    @staticmethod
#    def run_son(arg):
#        return arg.run_simulation()