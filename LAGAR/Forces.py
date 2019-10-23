# -*- coding: utf-8 -*-
"""


This module contains the different forces and processes that are going to be used in the kernels construction. 
More processes will be added in future releases.


.. sectionauthor:: Angel Daniel Garaboa Paz <angeldaniel.garaboa@usc.es>.


"""
import numpy as np


def AnisotropicDiffusion(r_p, D, Delta_t):
    """
    
    It computes the velocity due to diffusion.

    Args:
        - r_p (float): Position of the particle [space_units].
        - D (float): Diffusion rate or diffusion speed [space_units/time_units].
        - Delta_t (float): Timestep to consider the jump due to difusion [time_units].

    Returns:
        float: Difusion Velocity [space_units/time_units]


    """
    return (2.*np.random.random(r_p.shape)-1.) * np.sqrt(2.*D/Delta_t)


def MoralesDiffusion(r_p, D, Delta_t, v_f, w_f):
    """
    
    It computes the velocity due to diffusion.

    Args:
        - v_f (float): Velocity of the flow field at the particle position [m/s].
        - w_f (float): Velocity of the wf field at the particle position [m/s].
        - r_p (float): Position of the particle [space_units].
        - D (float): Diffusion rate or diffusion speed [space_units/time_units].
        - Delta_t (float): Timestep to consider the jump due to difusion [time_units].

    Returns:
        float: Difusion Velocity [space_units/time_units]


    """
    D = 10e-4*(272.8*np.abs(v_f) + 21.1*np.abs(w_f))
    return (2.*np.random.random(r_p.shape[1])-1.) * np.sqrt(2.*D/Delta_t)




def Stokes(v_p, v_f, rho_p, rho_f, nu_f, R):
    """
    This function computes the force exerted by viscosity when a perfect 
    sphere (without impurities) moves through a viscous fluid in a laminar flow.
    
    Args:
        - v_f (float): Velocity of the flow field at the particle position [m/s].
        - v_p (float): Velocity of the particle [m/s].
        - rho_p (float): Density of the particle [kg/m^3].
        - rho_f (float): Density of the fluid [kg/m^3].
        - nu_f (float): Kinematic viscosity of the fluid in [m^2/s].
        - R (float): Radius of the particle [m].
    
    Returns:
        -float: Stokes force on the particle [m/s^2]
    """
    return (-(9. * nu_f * rho_f) / (2.*rho_p*R**2.))*(v_p - v_f)



def Coriolis(r_p,v_p):
    """
    This function computes the force that acts on objects that are in motion
    relative to a rotating reference frame.
    
    Args:
        - r_p (float): Position of the particle [º].
    
    Returns:
        - float: Coriolis net force on the particle.
    """
    lat = r_p[:,1]
    return 2.*7.2921e-5*v_p*np.sin(lat[:,np.newaxis]*(np.pi/180.))



# def AddedMass(v_p, v_f, rho_f, rho_p):
#     """

#     This function computes the force exerted to the object due to 
#     the resistance of the fluid to be displaced by the particle
    
#     Args:
#         - v_p (float): Velocity of the particle [m/s].
#         - v_f (float): Velocity of the flow field at the particle position [m/s].
#         - rho_f (float): Density of the fluid [kg/m^3].
#         - rho_p (float): Density of the particle [kg/m^3].
    
#     Returns:
#         float: Added mass net force on the particle [m/s^2].
#     """
#     return -(rho_f/rho_p)*(v_p - v_f)


def Buoyancy(rho_p, rho_f):
    """
    This function computes the force exerted due to density diferences between
    the particle and the medium.
    
    Args:
        - rho_p (float): Density of the particle [kg/m^3].
        - rho_f (float): Density of the fluid [kg/m^3].
    
    Returns:
        - float: Buoyancy net force on the particle [m/s^2].
    """
    return np.array([[0.], [0.], [9.81*(1. - rho_p/rho_f)]])


def Windage(w_p, A, W):
    """
    Computes the influenced exerted by the wind flow over the floating particle and it is directly linked
    to the buoyancy of the object.
    
    Args:
        - w_p (float): Surface wind field at particle position [m/s].
        - A (float): Area of the particle above the surface  [m^2].
        - W (float): Area of the particle below the surface [m^2].
    
    Returns:
        - float: Velocity wind influence on the particle [m/s].
    
    """
    return w_p*0.015*np.sqrt(A/W) 



def Diffusion(r_p, v_f, w_f, D, Delta_t):
    """

    Computes the velocity due to diffuisión effect .
    
    Args:
        - r_p (float): Position of the particle [space_units].
        - v_f (float): Velocity of the flow field at the particle position [m/s].
        - w_f (TYPE): Desription.
        - D (float): Diffusion rate or diffusion speed [m/s] or [space_units/time_units].
        - Delta_t (float): Timestep to consider the jump due to difusion [s] or [time_units].
    
    Returns:
        - float: Difusion Velocity [m/s] or [space_units/time_units]

    """
    D_r = np.array([[D_f],[D_f],[(10.e-4)**(1+np.exp(1-10/r_p[2]))]])
    return (2.*np.random.random(r_p.shape[1])-1.)*np.sqrt(2.*D*Delta_t)/Delta_t


def TerminalVelocity(v_f,v_p,R,rho_f,rho_p,g,nu):
    """
    
    Args:
        v_f (float): Velocity of the flow field at the particle position [m/s].
        v_p (float): Velocity of the particle [m/s].
        R (float): Radius of the particle [m].
        rho_f (float): Density of the fluid [kg/m^3].
        rho_p (float): Density of the particle [kg/m^3].
        g (TYPE): Description
        nu (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    Re = np.abs(v_f-v_p)*R/nu_f
    
    if Re < 1:
        w = (R**2(rho_f-rho_p)*g)/(18.*nu_f)

    elif 1 < Re < 100:
        w = (0.223*R*((rho_f-rho_p)**2)*g**2)/(rho_f*nu_f)

    elif 100 < Re:
        w = 1.82 * ((rho_f-rho_p)*g*np.sqrt(R)/rho_f)

    return w

