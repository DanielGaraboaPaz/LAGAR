#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:18:43 2019

@author: daniel
"""

nPadzeros=5

import os

import numpy as np
import shutil
import xarray as xr
import os
import pandas as pd
import datetime
from pyevtk.hl import pointsToVTK 


class Writer:

		
        def __init__(self,initial_conditions):
            self.mem_or_disk = []
            self.extension = []
            self.path = []
            self.pattern = []
            self.writer_iterator = -1
            self.nPadzeros = 4
            self.debug = []
            self.initial_conditions = initial_conditions
            self.time_to_save = []
            self.files = []
            self.simulation_type=[]
            self.save_every_n_steps=[]

            

        def read_config(self, json_writer_dict):       
            self.mem_or_disk = json_writer_dict['mem_or_disk']
            self.extension = json_writer_dict['extension']
            self.path = json_writer_dict['path']
            self.pattern = json_writer_dict['pattern']
            self.simulation_type = json_writer_dict['simulation_type']

            if 'time_to_save' in json_writer_dict:
                if isinstance(self.time_to_save, float):
                     self.time_to_save =  json_writer_dict['time_to_save']
                elif len(self.time_to_save) == 3: 
                    self.time_to_save = np.arange(self.time_to_save[0],self.time_to_save[1],self.time_to_save[2])
                
            if 'save_every_n_steps' in json_writer_dict:
                self.save_every_n_steps = json_writer_dict['save_every_n_steps']

            return
        
              
        def allocate_solution(self,r0):
            self.mem_solution = np.zeros( (self.tspan.size,) + r0.shape)
            return		
            
        def create_path(self):
            if os.path.exists(self.path) == False and (self.writer_iterator == -1):
                try:
                    os.makedirs(self.path)
                except:
                    print('The directory exists')
                    return
            else:
                return
        
        def save_condition(self,t):
            if self.time_to_save:
                if np.any(np.abs((self.time_to_save - t[0])) == 0):
                    return True
                elif(self.time_to_save == 'all'):
                    return True
            if self.save_every_n_steps:
                if (self.writer_iterator % self.save_every_n_steps == 0):
                    return True
                
        
        def save_vtu(self,r,filename):
            x = np.ascontiguousarray(r[:,0], dtype=np.float32)
            y = np.ascontiguousarray(r[:,1], dtype=np.float32)
            if r.shape[1] == 2:
                z = np.zeros_like(x)
                pointsToVTK(filename[0:-4],x,y,z)
            elif r.shape[1] == 3:
                z = np.ascontiguousarray(r[:,2], dtype=np.float32)
                pointsToVTK(filename[0:-4],x,y,z)
            return
        
        
        def save_netcdf(self,r,t,filename):
            
            if self.simulation_type == 'emission':
                df = pd.DataFrame(r,columns=self.initial_conditions.names)
                df.index.name = 'id_particle'
                ds_step = df.to_xarray().expand_dims(dim='time')
                print('LAGAR_DEBUGGING__',t)  
                ds_step = ds_step.assign_coords(time = ds_step.time)
                ds_step.coords['time'].values = [t]
                reference_time = pd.datetime.strptime(self.initial_conditions.time_ref, '%Y-%m-%d %H:%M:%S')
                att_time_str = reference_time.strftime('X%Y-X%m-X%d X%H:X%M:X%S').replace('X0','X').replace('X','')
                ds_step.coords['time'].attrs = {'long_name':'time','units':'seconds since '+ att_time_str}
                ds_step.to_netcdf(filename)
                ds_step.close()
        
            elif self.simulation_type == 'grid':
                
                self.initial_conditions.ds.time.values = [t]
                self.initial_conditions.ds=self.initial_conditions.ds.assign_coords(time = self.initial_conditions.ds.time)
                reference_time = pd.datetime.strptime(self.initial_conditions.time_ref, '%Y-%m-%d %H:%M:%S')
                att_time_str = reference_time.strftime('X%Y-X%m-X%d X%H:X%M:X%S').replace('X0','X').replace('X','')
                self.initial_conditions.ds.time.attrs = {'long_name':'time','units':'seconds since '+ att_time_str}
                             
                if self.initial_conditions.mask.size:
                    k=0
                    for var in self.initial_conditions.names:
                        self.initial_conditions.ds[var].values.setflags(write=True)
                        self.initial_conditions.ds[var].isel(time=0).values[~self.initial_conditions.ds.mask]=r[:,k]
                        self.initial_conditions.ds[var].isel(time=0).values[self.initial_conditions.ds.mask]=np.nan
                        k=k+1
                        # Transpose the data before write to keep the order of the dimensions
                    transpose_keys = list(self.initial_conditions.ds.coords.keys())[::-1]
                    self.initial_conditions.ds.transpose(*transpose_keys).to_netcdf(filename)
                else:
                    k=0
                    for var in self.initial_conditions.names:
                        self.initial_conditions.ds[var].isel(time=0).values=r[:,k].reshape(self.initial_conditions.ds[var].shape)
                        self.initial_conditions.ds[var].isel(time=0).values[self.initial_conditions.ds.mask]=np.nan
                        k=k+1
                    
                    transpose_keys = list(self.initial_conditions.ds.coords.keys())[::-1]
                    self.initial_conditions.ds.transpose(*transpose_keys).to_netcdf(filename)
         
                
            if self.writer_iterator == self.initial_conditions.tspan.size-1:
                  #self.writer_iterator = 0
                  if '.nc' in self.extension:
                      
                    print('\n')
                    print('++++++++++++++++++++++++++++++++')
                    print('LAGAR Writer-Module information')
                    print('*+++++++++++++++++++++++++++++++')
                    print('Stage: Merging all nc-files into one dataset')
                    print('Saving file:', self.path + self.pattern+'final.nc','....')
                    
                    ds = xr.open_mfdataset(self.files,concat_dim='time',coords='minimal')
                    
                    ds.to_netcdf(self.path + self.pattern+'final.nc')
                    
                    print('Done')
                    print('\n')
        
        

        def save_step(self,r,t):
            
            self.writer_iterator = self.writer_iterator + 1

            if self.save_condition(t):

                if self.mem_or_disk == 'mem':
                    self.mem_solution[self.writer_iterator] = r
                	           
                	            
                elif self.mem_or_disk == 'disk':
                    filename = self.path + self.pattern + str(self.writer_iterator).zfill(self.nPadzeros) 
                    
                    print('\n')
                    print('++++++++++++++++++++++++++++++++\r')
                    print('LAGAR Writer-Module information\r')
                    print('*+++++++++++++++++++++++++++++++\r')
                    print('Saving file:', filename + self.extension)
                    print('Saving step:', self.writer_iterator)
                    print('Saving time:',t[0])
                    print('Number of particles:', str(r.shape[0]))
                    print('Number of variables:', str(r.shape[1]))
                    print('\n')
  
                    if '.bin' in self.extension:
                        filename_bin = filename + '.bin'
                        self.files.append(filename_bin)
                        r.tofile(filename_bin)
                    
                    if '.vtu' in self.extension:
                        filename_vtu = filename + '.vtu'
                        self.files.append(filename_vtu)
                        self.save_vtu(r,filename_vtu)
                        
                    if '.csv' in self.extension: 
                        filename_csv = filename + '.csv'
                        self.files.append(filename_csv)
                        np.savetxt(filename_csv,r, delimiter =';',fmt='%1.5f')
                        
                    if '.nc' in self.extension:
                        filename_nc = filename + '.nc'
                        self.files.append(filename_nc)
                        self.save_netcdf(r,t[0],filename_nc)
                        
                                  

        
            return