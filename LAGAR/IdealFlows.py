# -*- coding: utf-8 -*-
"""

This module contains ideal flows,
to use them as velocity fields inside the different kernels.

.. sectionauthor:: Angel Daniel Garaboa Paz <angeldaniel.garaboa@usc.es>
"""

import numpy as np

class ABC:

    """
    It generates a ABC flow object.

    The Arnold–Beltrami–Childress (ABC) flow is a three-dimensional 
    incompressible velocity field which is an exact solution of Euler's equation.

    Attributes:

        - A (float): A parameter.

        - B (float): B parameter.

        - C (float): C parameter.

    """

    
    def __init__(self, A, B, C):
        
        """
        
        Args:
            A (float): A parameter.

            B (float): B parameter.

            C (float): C parameter.

        """
        self.A = A
        self.B = B
        self.C = C


    def F(self,r,t):
        """

        This method evaluates the velocity at the given postion.
        
        Args:
            r (float): Particle position.

            t (float): time (not needed).
        
        Returns:
            float: Velocity of the flow field at the particle postion r
        
        """
        return np.c_[self.A * np.sin(r[:,2]) + self.C*np.cos(r[:,1]),
                         self.B * np.sin(r[:,0]) + self.C*np.cos(r[:,2]),
                         self.C * np.sin(r[:,1]) + self.B*np.cos(r[:,0])]


class DoubleGyre:

    """
    It generates a two-dimensional flow consisting of two vortices rotating in a conterwise way inside 
    a box with dimensions x=[0,2] and y=[0,1]
    
    Attributes:
        A (float): Factor to control the magnitude of the velocity.

        eps (float): Factor to control the how far the line separating the gyres moves to the left or right.

        omega (float): frequency of oscillation.
    """
    
    def __init__(self, omega, eps, A):
        """
        
        Args:
            A (float): Factor to control the magnitude of the velocity.
            eps (float): Factor to control the how far the line separating the gyres moves to the left or right.
            omega (float): Frequency of oscillation.
        """
        self.omega = omega
        self.eps = eps
        self.A = A

    def F(self, r, t):
        """
        This method evaluates the velocity at the given postion.
        
        Args:
            r (float): Particle position.

            t (float): Time (not needed).
        
        Returns:
            float: Velocity of the flow field at the particle postion r.
        """
        a = self.eps*np.sin(self.omega*t)
        b = 1.-2.*self.eps*np.sin(self.omega*t)
        f = a*r[0]**2.+b*r[0]
        dfdx = 2.*a*r[0]+b
        return np.stack((-np.pi*self.A*np.sin(np.pi*f)*np.cos(np.pi*r[1]),
                         np.pi*self.A*np.cos(np.pi*f)*np.sin(np.pi*r[1])*dfdx))


class Lorenz:

    """
    The Lorenz model is simplified mathematical model for atmospheric convection
    widely used to observe the chaotic behavior.
    
    Attributes:
        A (float): Parameter proportional to Prandtl number.

        B (float): Parameter proportional to Raileygh number.

        C (float): Physical dimension of the layer.
    """
    
    def __init__(self, A, B, C):
        """
        
        Args:
            A (float): Parameter proportional to Prandtl number.

            B (float): Parameter proportional to Raileygh number.

            C (float): Physical dimension of the layer.
        """
        self.A = A
        self.B = B
        self.C = C

    def F(self, r, t):
        """
        This method evaluates the velocity at the given postion.
        
        Args:
            r (float): Particle position.
            
            t (float): Time (not needed).
        
        Returns:
            float: Velocity of the flow field at the particle postion r.
        """
        return np.stack((self.A*(r[1]-r[0]), r[0]*(self.B-r[2])-r[1], r[0]*r[1]-self.C*r[2]))

