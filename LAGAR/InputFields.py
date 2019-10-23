#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:12:31 2019

@author: daniel
"""
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator,griddata
from interpolation.splines import LinearSpline
import numpy as np



# to install LinearSpline interpolator conda install -c conda-forge LinearSpline

class InputFields:
    """
    This module reads NetCDF files from local disk or online thredds servers through OPENDAP
    protocol using the xarray module.

    Attributes:
        - ds (xr.Dataset): xarray dataset format of the NetCDF, with the dimensions ordered and
        transported according to JSON file "dims_order" and "dims_dataset" values.

        - grid (list): Get the grid structure of the dataset and turns it into numerical
        values in order to build up the interpolant functions.

        - interpolants (list): List of the interpolant functions of the variables in the "fields" JSON file.  
    """

    def __init__(self):
        """
        Get the information from the "input" field in the JSON file.

        Args:
            **input: The dictionary containing the information regarding the input setup in the JSON file.
        """
        
        self.field_name = []
        self.file_name = []
        self.dims_dataset = []
        self.dims_fields = []
        self.dims_order = []
        self.dims_drop = []
        self.fields = []
        self.conversion_grid = []
        self.origin_coords = []
        self.origin_points = []
        self.destiny_coords = []
        self.destiny_points = []
        self.interpolants = []
        self.grid = []
        self.ds = []
        self.dims_slice = []
        self.reference_time = []
        self.numba_interpolant=[]
        
    def get_info(self,json_dict):
        self.file_name = json_dict['file_name']
        self.dims_dataset = json_dict['dims_dataset']
        self.dims_fields = json_dict['dims_fields']
        self.dims_order = json_dict['dims_order']
        self.fields = json_dict['fields']
        
        if 'reference_time' in json_dict:
            self.reference_time = json_dict['reference_time']
        
        if 'dims_drop' in json_dict:
            self.dims_drop = json_dict['dims_drop']
        
        if 'dims_slice' in json_dict:
            self.dims_slice = json_dict['dims_slice']
            
        if 'conversion_grid' in json_dict:
            self.conversion_grid = json_dict['conversion_grid']
            self.origin_coords = json_dict['origin_coords']
            self.destiny_coords = json_dict['destiny_coords']
        
        if 'numba_interpolant' in json_dict:
            self.numba_interpolant = json_dict['numba_interpolant']
        else:
            self.numba_interpolant = 0
            
        return
    
        
        

    def get_mfds(self):
        """
        Reads a ordered list of netcdf, with the dimensions ordered and 
        transposed according to JSON file "dims_order" and "dims_dataset" values.
        
        Returns:
           xarray Dataset header
        """
        self.ds = xr.open_mfdataset(self.file_name)
        
        # Remove duplicates inside the dataset
        for key in self.dims_dataset:
            np.diff(self.ds[key])
            _,index, counts = np.unique(self.ds[key], return_index=True,return_counts=True)
            if np.any(counts>1):
                self.ds = self.ds.isel({key:index})
                print(f"\033[7;1;5;{'31m'}{'WARNING:Your input data contains repeated values dimension values.'}\033[00m")
                print(f"\033[7;1;5;{'31m'}{'I will try to remove those values but the results could be inacuratte.'}\033[00m")
                print(f"\033[7;1;5;{'31m'}{'Do not use numba_interpolants with fixed step under this situation:WARNING.'}\033[00m")
            
        slice_dict = {}
        for key in self.dims_slice:
                slice_0 = self.ds[key].sel({key:self.dims_slice[key][0]},method='ffill').values
                slice_1 = self.ds[key].sel({key:self.dims_slice[key][1]},method='bfill').values
                slice_dict[key] = slice(slice_0,slice_1, self.dims_slice[key][2])


            

        if self.dims_drop:
            self.ds = self.ds.sel(slice_dict).drop(self.dims_drop).squeeze()
                
        elif self.dims_slice:
            self.ds = self.ds.sel(slice_dict)

            
            
            
        dim_to_order = [dim[0] for dim in zip(
            self.dims_dataset, self.dims_order) if dim[1] == -1]
        
        if dim_to_order == []:  
            self.ds = self.ds.transpose(*self.dims_dataset)
        else:
            self.ds = self.ds.transpose(*self.dims_dataset).sortby(dim_to_order, ascending=True)
        
        print('+++++++++++++++++++++++++++++++')
        print('LAGAR InPut dataset information')
        print('+++++++++++++++++++++++++++++++')
        print('File readed:',self.file_name)
        for keys in self.ds.dims:
            print(keys,'=>','Min:',self.ds[keys].min().values,'Max:',self.ds[keys].max().values)
        
#        if len(self.dims_slice) > 0:
#            print(self.dims_slice)
#           
#            check_integers = []
#            for slices in self.dims_slice:
#                for dim in slices:
#                    check_integers.append(isinstance(dim,int))
#                
#            if all(check_integers):
#                print('Suppossed integers',self.dims_slice, check_integers)
#                self.dims_slice = dict(zip(self.dims_dataset,map(slice,*self.dims_slice)))
#                self.ds = self.ds.isel(self.dims_slice)
#            else:
#                self.dims_slice = dict(zip(self.dims_dataset,map(slice,*self.dims_slice)))
#                self.ds = self.ds.sel(self.dims_slice)

        self.ds = self.ds.fillna(0)

       
    def get_ds(self):
        """
        Reads the NetCDFs, with the dimensions ordered and 
        transposed according to JSON file "dims_order" and "dims_dataset" values.
        
        Returns:
           xarray Dataset header
        """

        dim_to_order = [dim[0] for dim in zip(
            self.dims_dataset, self.dims_order) if dim[1] == -1]
            
        self.ds = xr.open_dataset(self.file_name).transpose(
            *self.dims_dataset).sortby(dim_to_order, ascending=True).fillna(0)
        return


    def get_grid(self):
        """
        """
        for grid_name in self.dims_fields:
            if (grid_name == 'time') and (self.ds['time'].dtype != 'float64'):
                if self.reference_time:
                    reference_time = np.datetime64(pd.datetime.strptime(self.reference_time, '%Y-%m-%d %H:%M:%S'))
                    self.grid.append((self.ds[grid_name].values -reference_time).astype('timedelta64[s]').astype('f8'))
                else:
                    reference_time = self.ds[grid_name].values[0]
                    self.grid.append((self.ds[grid_name].values -reference_time).astype('timedelta64[s]').astype('f8'))
            else:
                self.grid.append(self.ds[grid_name].values)

#        print('+++++++++++++++++++++++++++++++')
#        print('LAGAR InPut dataset information')
#        print('+++++++++++++++++++++++++++++++')
#        print('Interpolant grids:',self.dims_name)
#        for keys in self.dims_fields:
#            print(keys,'=>','Min:',self.ds[keys].min().values,'Max:',self.ds[keys].max().values)
        return

    def get_interpolants(self, method='linear'):
        """

        Get the interpolant functions using the scipy engine or the interpolation.splines engine.
        The second option requires the installation of this package with pip.
        It provides a very faster interpolation algorithm, however, the data points 
        should be regularly spaced. The scipy is more flexible, allowing not equally spaced 
        points in a dimension, but it is slower.


        Args:
            - numba_interpolant (bool, optional): Select scipy or interpolation engine.
            - method (str, optional): Nearest, linear, spline.

        """
        if self.conversion_grid == 1:
               self.origin_points = np.column_stack(([self.ds[name].values.ravel() for name in self.origin_coords]))
               self.destiny_points = map(np.ravel,np.meshgrid(*[self.ds[name].values.ravel() for name in self.destiny_coords]))
               # self. = np.column_stack(([mesh.ravel() for mesh in destiny_mesh]))
                       
        if self.numba_interpolant == 0:
            for field in self.fields:
                self.interpolants.append(RegularGridInterpolator(
                    self.grid, self.ds[field].values, method, bounds_error=False))

        elif self.numba_interpolant == 1:
            for field in self.fields:
                self.interpolants.append(LinearSpline(list(map(np.min, self.grid)), list(map(
                    np.max, self.grid)), list(map(np.size, self.grid)), self.ds[field].values))
        

       
        return

    def update_interpolant(self,time_slice):
        self.get_mfds(time_slice=time_slice)
        self.get_grid()
        self.get_interpolants()
        return
        

    def F_s(self,r):
        """
        Evaluates the static function using the previously computed interpolants.
        It can be a scalar or vectorial function.
        
        Args:
            - r (array): np.array with the array of points
            - t (float): Time.

        Returns:
            - array: Value the funtion evaluated at F(r,t).
        """

        return np.stack([f_i(r) for f_i in self.interpolants],axis=1)    
    
        
    def F(self,r,t):
        """
        Evaluates the function using the previously computed interpolants.
        It can be a scalar or vectorial function.
        
        Args:
            - r (array): np.array with the array of points
            - t (float): Time.

        Returns:
            - array: Value the funtion evaluated at F(r,t).
        """

        if self.conversion_grid == 1: 
            r = np.column_stack(([griddata(self.origin_points,destiny_coords,r) for destiny_coords in self.destiny_points]))

        r_t = np.c_[r, t]
        

        return np.stack([f_i(r_t) for f_i in self.interpolants],axis=1)


#F(r,t):
#        
#    r = [xr.DataArray(np.squeeze(r[:,i]),dims=['z']) for i in range(0,r.shape[1])] + t
#    r_to_interpolate = 

class MultipleInputFields:
    
    def __init__(self,input_fields):
        self.input_fields = input_fields
        self.mask = []
    
    
    
    def check_domains(self,r):
        self.mask = np.ones(r.shape[0],len(self.input_fields))
        for domain in range(0,len(self.input_fields)):
            for dim in range(0,r.shape[1]):
                self.mask[:,domain] = self.mask[:,domain]*self.input_field.grid[dim][0] < r[:,dim] < self.input_field.grid[dim][-1]
        
        
        for domain in range(0,len(self.input_fields)):
            self.mask[self.mask[:,domain],domain+1] = False
    
    def F(self,r,t):
        """
        Evaluates the function using the previously computed interpolants.
        It can be a scalar or vectorial function.
        
        Args:
            - r (array): np.array with the array of points
            - t (float): Time.

        Returns:
            - array: Value the funtion evaluated at F(r,t).
        """

        r_t = np.c_[r, t]
        v_t = np.zeros_like(r)
        
        np.stack([f_i(r_t) for f_i in self.interpolants],axis=1)
        return 
           
        
        
        
        
        
        
        
    
    