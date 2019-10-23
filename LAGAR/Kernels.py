# -*- coding: utf-8 -*-
"""

This module contains different kernels or macro F functions from drdt=F(r(t),t)
ODE systems. The idea of these Kernels is to encapsulate inside an object all
the parameters and functions required to evaluate the particle properties
that r(t) involve in time, in such a way that finally we only get a function F,
where the only parameters needed to evaluate are the time-dependent ones.
This allows us, to send this function F to be evaluated with the Solver package.
New kernels containing different properties will be added in the future.


.. sectionauthor:: Angel Daniel Garaboa Paz <angeldaniel.garaboa@usc.es>

"""
import numpy as np
from LAGAR import Forces
from LAGAR.InputFields import InputFields
import xarray as xr



class Lagrangian:

    """
    This class provides pure Lagrangian kernels. 
    The particle just follows the local velocity flow field.


    Attributes:
        -Vi (function): The interpolant function, to evaluate the
        local flow velocity field V(r(t),t).
    """

    def __init__(self,json_dict):
        """
        Lagrangian kernel constructor.

        Args:
            Vi (function): Interpolant function.
        """
        
        self.Vi = InputFields()
        self.Vi.get_info(json_dict['velocity'])
        self.Vi.get_mfds()
        self.Vi.get_grid()
        self.Vi.get_interpolants()
        
        self.boundaries = []


        

    def F(self, r, t):
        """

        Model Kernel function to send to the solver module.

        Args:
            r (array): Array containing the position of the particles [space_units].
            t (float): Time instant [time_units]

        Returns:
            array: Array containing the velocity componentes evaluted. [space_units/time_units]
        """
        if self.boundaries:
                r = self.boundaries.F(r)

        
        return self.Vi.F(r, t)


class LagrangianSpherical2D:

    """
    This class provides Lagrangian Kernels, for spherical (lat,lon) integrations,
    considering a perfectly spherical earth with a 6370Km radius.
    The particle just follows the local flow velocity field.

    Attributes:
        - m_to_deg (float): Conversion from [m/s] to [degrees/s].
        - Vi (function):  The interpolant function, to evaluate the local flow velocity field V(r(t),t).
    """

    def __init__(self, json_dict):
        """
        Lagrangian kernel constructor.

        Args:
            Vi (function): Interpolant function.
        """
        self.m_to_deg = (np.pi/180.)*6370000.
        self.Vi = InputFields()
        self.Vi.get_info(json_dict['velocity'])
        self.Vi.get_mfds()
        self.Vi.get_grid()
        self.Vi.get_interpolants()
        

    def F(self, r, t):
        """

        Model Kernel function to send to the solver module.

        Args:
            - r (array): Array containing the position of the particles [space_units].
            - t (float): Time instant [time_units].

        Returns:
            - array: Array containing the velocity componentes evaluted [space_units/time_units].

        """
        
        drdt = self.Vi.F(r, t)
        drdt[:,0] = drdt[:,0]/(self.m_to_deg*np.cos((np.pi/180.)*r[:,1]))
        drdt[:,1] = drdt[:,1]/self.m_to_deg

        
        return drdt
