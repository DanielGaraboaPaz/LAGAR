LAGAR is the name of a python package to perform LAGrangian Analysis and Research of particle trajec-
tories using analytical and/or real flow fields. This package was developed at the Non-Linear Physics
Group in the University of Santiago the Compostela (USC), Spain by Angel Daniel Garaboa Paz and Prof.
Vicente Perez Muñuzuri,
This package is intended to solve the transport problem using a dynamical systems approach,
d r /dt = F ( r (t),t)
with initial conditions
r (t_0)= r _0.
1.2 why-LAGAR
LAGAR is a python package intended to study the transport problem using a dynamical systems
approach without being a particular solver for a given predefined particle. The idea is to create a base
code to create F functions and this functions can be evaluated using real flow data in a user friendly way
Their main purpose and advantages are:
• we solve ordinary differential equations (hereafter, ODES) problems using python:,
We rewrite the evaluations of F function, into encapsulated Kernels objects. They containing all the infor-
mation regarding to other subfunctions or parameters, to evaluate F ( r (t),t).
• It evaluates at once arrays of millions of initial conditions using the interpolants for real data (scipy
or interpolate packages) increasing the speed up in many orders of magnitude, avoiding map or
vectorial functions from numpy which solves each initial conditions individually.
3LAGAR Documentation, Release Alpha 0.1
• The build of the interpolants to evaluate F is an automated process hidden for the user.
• The setup of the problem for just require to define one external JSON file, allowing the scripting of
the code to run it automatically for operational purposes, or to run parametric analysis.
• The extension to include new particles to test the effects on the solutions into new Kernel functions it
is really easy to done.
• It can write outputs in multiple file formats: netcdf, csv, vtu.
• It includes a postprocessing module to compute different measures from trajectories or solutions
such as FTLE, or residence times.
