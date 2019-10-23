# -*- coding: utf-8 -*-
"""

This module contains functinality to read the setup of the input data
and initial conditions.


.. sectionauthor:: Angel Daniel Garaboa Paz <angeldaniel.garaboa@usc.es>

"""
import numpy as np
import pandas as pd
from LAGAR.InputFields import InputFields
import xarray as xr
#from interpolation.splines import CubicSpline, LinearSpline


          
class InitialConditions:

    """
    This module creates regular meshes of multidimensional initial conditions,
    and turn it into arrays of points, using the order: [variable, solution_id].
    
    Example: Given 10 Lagrangian particles in cartesian coordinates. we require x,y,z position
    to describe its state at t time, so the resulting array will be: [3,10]. 

    Using this,
    ::
        r0[1]

    we take the x-component for all the particles.

    Also, if a mask is provided, the points can be masked and suppressed in
    order to create regular initial conditions with any shape considered.


    Future: The idea is to increase its functionality by adding new functions for 
    multidimensional problems and non-regular geometries.

    Attributes:
        - r0 (np.array): An array of initial points.
        - r0_raw (list): Copy of the original array with a mask included, to
        - create a np.masked array.
        - mask (bool): Array of booleans to select the valid points to compute.
        - span (np.array): The initial grid of points to generate the initial conditions domain.
    """

    def __init__(self):
        """
        Init the object to create the initial conditions.

        Args:
            - **r0_dict: The information provided by the dictionary "r0" in the setup JSON file.
        """
        self.setup = []
        self.names = []
        self.ranges=[]
        self.step = []
        self.span = []
        self.mask = []
        self.maski = []
        self.r0 = []
        self.r0_raw = []
        self.t0 = []
        self.time_range = []
        self.dt = []
        self.r = []
        self.t = []
        self.t_fmt = []
        self.t0_fmt = []
        self.dt_fmt = []
        self.ds = []
        self.coords_names =[]
        self.tspan = []
        self.tspan_fmt = []
        self.parameters = []



    def read_config(self,json_dict):
        self.names = json_dict['names']
        if 'file' in json_dict:
            self.file=json_dict['file']
        else:
            self.ranges = json_dict['ranges']
            self.step = json_dict['step']
        self.time_range = json_dict['time_range']
        if 'reference_time' in json_dict:
            self.time_ref = json_dict['reference_time']
        if 'parameters' in json_dict:
            self.parameters = json_dict['parameters']
        
        return

        

    def spanning(self):
        """
        
        Turns the information provided by the by the dictionary "r0" in the setup JSON file,  
        regarding variables into arrays of initial conditions to define the initial mesh axis. 

        Returns:
            - list: List of np.arrays.
        """

        for k in range(0, len(self.names)):
            self.span.append(
                np.arange(self.ranges[k][0], self.ranges[k][1], self.step[k]))

        
        return

    def set_array(self):
        """
        
        It creates the array of initial conditions.


        Returns:
            - array: np.array of initial conditions in the form [varible, particle_id]

        """
        
        mesh = np.meshgrid(*self.span,indexing='ij')
        # here we use indexing ij, this is to work with the correspondence
        # x,y,z --> i,j,k indexing. Then, when the netcdf is writen, this is
        # tranposed in order to keep the netcdf convection (k,j,i indexing)
        
        self.coords_names = [name + "_0" for name in self.names]
        nvars = len(self.names)
        coords = dict(zip(self.coords_names, zip(self.coords_names,self.span)))
         
        variables = dict(zip(self.names, zip(nvars*[self.coords_names],mesh)))
        self.ds = xr.Dataset(variables,coords=coords)
        self.ds = self.ds.expand_dims(dim='time',axis=len(self.coords_names))
        
        for var in self.names:
            self.ds[var].values.setflags(write=True)
            
        self.r0 = np.stack((list(map(np.ravel, mesh))))
        self.r0 = self.r0.transpose()
        
        return
    
    def set_mask(self,json_dict):
        self.maski = InputFields()
        self.maski.get_info(json_dict['mask'])
        self.maski.get_mfds()
        self.maski.get_grid()
        self.maski.get_interpolants(method='nearest')
        

        

    def apply_mask(self):
        """

        Apply the mask to the points, and filter the non-true points.
        It also generates a copy of the array with the mask included (numpy masked array) 
        to reconstruct the initial array, in the postprocessing stage. 

        WARNING:: 
            The masked array cannot be passed directly to computations (it is heavily slow).
    

        """
        print('\n')
        print('*************************************')
        print('LAGAR InitialConditions Information ')
        print('*************************************')
        print('Info: True condition => Point masked by position')
        print('Info: Points True condition => not integrate')
        print('Points pre-mask:',self.r0.shape[0])
        print('Stage: Appliying mask filter points')
               
        self.mask = self.maski.F_s(self.r0) == 1 
        # Added tuple to avoid the Future Warning.    
        remove_mask = (~self.mask).any(axis=1)          
        self.r0 = self.r0[(remove_mask)]
        
        if self.ds: 
            self.ds['mask'] = (self.coords_names,self.mask.reshape(self.get_dims()))
 
        print('Points post-mask:',self.r0.shape[0])   
        print('\n')
        
        return

    def read_points_file(self, delimiter=';'):

        """
        Read the information of initial conditions from a CSV file.
        
        Args:
            delimiter (str, optional): delimiter for CSV files.
        
        Returns:
            void: sets the array with the CSV readed values.
        
        """
        
        
        if self.file[-4:] == '.csv':
            csv = pd.read_csv(self.file, delimiter=delimiter)
            self.r0 = csv[self.names].values
        
        elif self.file[-3:] == '.nc':
            ds = xr.open_dataset(self.file)
            try:
                self.r0 = np.stack([ds[name].values for name in self.names],axis=1)  
            except:
                self.r0 = np.array([ds[name].values for name in self.names])
        return

        
    def span_time(self):

        if isinstance(self.time_range[0],str):
            self.tspan_fmt = pd.date_range(start=self.time_range[0],end=self.time_range[1],freq=self.time_range[2])
            self.reference_time = pd.datetime.strptime(self.time_ref, '%Y-%m-%d %H:%M:%S')
            self.dt_fmt = self.time_range[2]
            self.t0_fmt = self.tspan_fmt[0]
            self.tspan = (self.tspan_fmt-self.reference_time).astype('timedelta64[s]').astype('f8').values
            self.t0 = self.tspan[0]
            self.dt = float(self.time_range[2][0:-1])
            
            print('+++++++++++++++++++++++++++++++')
            print('LAGAR InitialCondtions time')
            print('+++++++++++++++++++++++++++++++')
            print('Reference_time:',self.reference_time)
            print('Number of steps:',self.tspan)
            print('Time step used:',self.dt)
            print('Date started:', self.t0_fmt)
            print('Date end:',self.tspan_fmt[-1])
            print('Date started in seconds', self.t0)
            print('Date end in seconds:',self.tspan_fmt[-1])           
                       
        else:
            
            self.tspan = np.arange(self.time_range[0],self.time_range[1],self.time_range[2])
            self.t0 = self.tspan[0]
            self.dt = self.time_range[2]
            
            print('+++++++++++++++++++++++++++++++')
            print('LAGAR InitialCondtions time')
            print('+++++++++++++++++++++++++++++++')
            print('Number of steps:',self.tspan.size)
            print('Time step used:',self.dt)
            print('Date started in seconds', self.t0)

        
        return

    def get_dims(self):
        """

        Get the dimensions of the initial conditions domain.

        Returns:
            list: the list containing the dimensions of the initial conditions domain.
        """

        return list(map(np.size, self.span))


    def generate(self,json_dict):
        self.read_config(json_dict)
        if 'file' in json_dict:
            self.read_points_file()
        else:
            self.spanning()
            self.set_array()
        if 'mask' in json_dict:
            self.set_mask(json_dict)
            self.apply_mask()
        self.span_time()
    
    def update(self,json_dict):
        self.read_config(json_dict)
        self.span_time()
                
        
        

class Emissors:
    
    def __init__(self,):
        self.ds = []
        self.emissor_flag = 0
        self.tspan=[]

    
    def read_config(self,json_dict):
        self.ds = xr.open_dataset(json_dict['file_input'])
        self.emission_vars = json_dict['emission_vars']
        self.reference_time = np.datetime64(pd.datetime.strptime(json_dict['reference_time'], '%Y-%m-%d %H:%M:%S'))
        self.tspan = (self.ds.time.values-self.reference_time).astype('timedelta64[s]').astype('f8')
        return
    
    
    def emission(self,r,t):

        if np.any(np.abs((self.tspan - t[0])) == 0):
            time_index = np.where(np.abs((self.tspan - t[0])) == 0)[0]
            
            ds_emision = self.ds.isel(time=time_index[0])
            
            for emisor_id in range(0,ds_emision.emisor_id.size):
                particle = np.stack([ds_emision[var].isel(emisor_id = emisor_id).values for var in self.emission_vars])
                number_of_particles = np.int(ds_emision.isel(emisor_id = emisor_id).rate.values)
                new_particles = np.tile(particle,(number_of_particles,1))
                
                r = np.r_[r,new_particles]
                
            t = np.tile(t[0],r.shape[0])
        return r,t


    
#class Emissors:
#    
#    def __init__(self,initial_conditions):
#        self.initial_conditions = initial_conditions
#        self.emission_rate = []
#        self.emission_time = []
#        
#    
#    def read_config(self,json_dict):
#        self.emission_rate = json_dict['emission_rate']
#        self.emission_time = json_dict['emission_time']
#        self.emission_time = np.arange(self.emission_time[0],self.emission_time[1],self.emission_time[2])
#        return
#    
#    
#    def emission(self,r,t):
#        
#        if np.any(np.abs((self.emission_time - t[0])) == 0):
#            id_emissor = 0
#            for emissor in range(0,self.initial_conditions.r0.shape[0]):
#                new_particles = np.tile(self.initial_conditions.r0[id_emissor,:],(self.emission_rate,1))
#                r = np.r_[r,new_particles]
#                
#                id_emissor = id_emissor +1
#            t = np.tile(t[0],r.shape[0])
#        return r,t
        
        