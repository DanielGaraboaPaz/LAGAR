.. highlight:: rst

LAGAR summary
=============

LAGAR is python package to address the transport problem using a dynamical systems approach, 

d **r** /dt = **F** ( **r** (t),t)

with initial conditions

**r** (t_0)= **r** _0.

We seek for an approach where the vector **r**\(t) represents the state of the particle. In case of particle trajectories and in the easiest case this vector represents the position of the particle in 2D or 3D. The LAGAR package, extend this concept to n-dimensional problems and the state vector
of the particle can carry out any number of properties, making the problem extendable to a n-dimensional.
The **F** functions represents the dynamical system changing the state vector on time. In the simplest case, this function is the flow field, changing only the position. In the same way, that we address the state vector, the **F** function can be any function writed by the user and extendable
to any n-dimensional system. 


why-LAGAR
=========

LAGAR is a "bridge" between the transport problem and being user-friendly with a low python knowledge. It is intended to avoid the hardest part of coding and solving the ordinary diferential equations in case of real flow fields, to provide and user friendly way to create you own **F** functions and your own **r** state vectors and evaluate them using real data. Their key functionalities or advantages are:

- The evaluations of **F** function in case you are using real flow data, are done into encapsulated the called Kernels objects. They containing all the information regarding to other subfunctions or parameters, to evaluate **F** ( **r** (t),t). The build of the interpolants to evaluate the **F** in case of real data fields is an automated process hidden for the user.

- The different inial conditions are done in one unique array of **n** x **m** (initial conditions x variables). It evaluates **F** function at once. This allow to evaluate arrays of millions of initial conditions using the interpolants for real data (scipy or interpolate packages) increasing the speed up in many orders of magnitude in compare with map, vectorial functions or other scipy.odesolver which solves each initial conditions individually.

- The setup of the problem and their inputs, just require a JSON file, allowing the scripting of the code to run it with operational purposes or parametric analysis purporses.

- To add new particles (new **F** functions) to test their effects on the solutions can be done in a friendly way.

- It can write outputs in multiple file formats: netcdf, csv, vtu.

- It includes a postprocessing module to compute different measures from trajectories or solutions such as FTLE, or residence times.