# -*- coding: utf-8 -*-
"""

This module contains different ODE solvers.
The idea is to keep the same input structure than
others ODE solvers from scipy packages.  
More solvers will be added in the future.

.. sectionauthor:: Angel Daniel Garaboa Paz <angeldaniel.garaboa@usc.es>

"""
import numpy as np
import datetime
import time
from tqdm import tqdm 


class Solvers:
    
    def __init__(self, initial_condition, kernel, writer, boundaries, emissor):
        self.initial_condition = initial_condition
        self.kernel = kernel
        self.writer = writer
        self.emissor = emissor
        self.boundaries = boundaries
        self.chunk_integration = 0
        self.solver_type = []

    def RK4_step(self,r,t,dt):
        """
        A fixed step Runge-Kutta 4 order solver.
        
        Args:
            - f (function): Kernel-ODE to evaluate.
            - tspan (float): Time steps to solve the ODE.
            - r0 (float): Initial conditions.
        
        Returns:
            - r,t: The solution at each point evaluated every timestep.
        """
        
        k1 = self.kernel.F(r, t)
        k2 = self.kernel.F(r + k1*dt/2., t + dt/2.)
        k3 = self.kernel.F(r + k2*dt/2., t + dt/2.)
        k4 = self.kernel.F(r + k3*dt, t + dt)
        
        r = r + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        t = t + dt
        return r,t

    def Euler_step(self,r,t,dt):
        """
        A fixed step Euler ODE solver.
        
        Args:
            - self (function): Kernel ODE function to solve.
            - t (float): Time to evaluate t to solve the ODE.
            - r0 (float): Initial conditions.
        
        Returns:
            - r,t: The solution at t+dt evaluated at time t.
        """
        
        r = r + self.kernel.F(r, t)*dt
        t = t + dt
        return r,t


    def print_module_info(self,r,t):
        print('\n')
        print('+++++++++++++++++++++++++++++++\r')
        print('LAGAR Solver-Module information\r')
        print('+++++++++++++++++++++++++++++++\r')
        print('Time saved:', t[0],'\t', )
        print('Points:',r.shape[0],'\r')
        return


    def Integrate(self):
        """
        Main integration function.
        
        Args:
        - self (solver): Kernel ODE function to solve.
        - tspan (float): Time steps to solve the ODE.
        - r0 (float): Initial conditions.
        
        Returns:
        - array: The solution at each point evaluated every timestep.
        """
        
        k = 0
        self.writer.iterator = 0
        self.writer.create_path()
        
        r = self.initial_condition.r0.copy()
        t = np.full(r.shape[0],self.initial_condition.t0)
        dt = self.initial_condition.dt
        nsteps = self.initial_condition.tspan.size
        
        for k in tqdm(range(0,nsteps), desc='Percentaje completed:\r'):
        
            self.writer.save_step(r,t)
            if self.solver_type == 'RK4':
                	r,t = self.RK4_step(r,t,dt)
            elif self.solver_type == 'Euler':
                	r,t = self.Euler_step(r,t,dt)

            if self.emissor:
                r,t = self.emissor.emission(r,t)
            
            # return the object ready for a next integration.
            self.initial_condition.r0 = r
            self.initial_condition.t0 = t
            if (isinstance(t,datetime.datetime)):
                self.initial_condition.t_fmt = self.initial_condition.t0_fmt + datetime.timedelta(seconds=t[0])
            
        return

           