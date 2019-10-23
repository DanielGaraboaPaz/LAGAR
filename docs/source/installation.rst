============
Installation
============

At this moment LAGAR does not contain an automatic way to be installed. 
We are currently working on it. Once it is finished it should be installed using:

pip install git+https://github.com/DanielGaraboaPaz/LAGAR.git

At this development stage, it can only be installed manually (by the "hard" way). You can download  the master.zip from the github reporsitory: 

.. _LAGAR: https://github.com/DanielGaraboaPaz/LAGAR/archive/master.zip

unzip it and add the LAGAR folder to the PYTHONPATH manually.

If you have access to tar.gz build. You can install it with:

::

	$ pip install LAGAR-0.1.tar.gz


Please check that dependencies are satisfied. We strongly recomend the use of pip and Anaconda to install them:
- numpy
- xarray
- scipy
- pandas
- interpolate (from conda-forge interpolate, to use Numba Interpolants)
- scikit-image (in your are going to use postprocessing)
- json
- tqdm

================
how to run LAGAR 
================

To run LAGAR, once you installed run the following command:

::

   $ python -m LAGAR LAGAR_setup.json 1

The LAGAR has two inline input arguments, the first one is the json setup. The second one, is optional and control the number of chunks to divide the simulation in case
that you want to perform longs integration with large data fields. (This will be deprecated in future releases) 