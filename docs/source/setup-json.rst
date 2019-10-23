.. highlight:: rst

Setup Init
==========

Core of the LAGAR package. This init script reads and connects the different inputs and outputs of the different modules. It works as a template and it can be modified.


Reading the config file
-----------------------

It reads the JSON file described in `LAGAR Setup`_ passed as a argument from the command line,

::

    $ python LAGAR_init.py LAGAR_setup.json


Setup json template
===================


Before running, the first step is to configure the :file:`LAGAR_setup.json`. This file contains different
dictionaries to be read by the LAGAR main method. Each dictionary inside the JSON file configures a module.



Initial Conditions
------------------

.. inheritance-diagram:: LAGAR.InitialConditions
   :parts: 1


The first dictionary controls the creation of initial conditions by :func:`LAGAR.InitialConditions.InitialConditions`:

::

    "r0":{
    "names":["lon","lat","depth"],
    "ranges":[[-8.95,-8.65],[42.10,42.40],[8,9]],
    "step":[0.01,0.01,1]
    },

The keyword :term:`r0', controls de initial conditions for the differentes variable names
:term: 'names'. The :term:`ranges', specify the limits over each variable name in order
to create a set of points with the resolution specified in :term:`step'. This, generates a
square, cube or hipercube of points of initial conditions.


Inputs
------
The input dictionary controls the reading of external files with :func:`LAGAR.InputFields`. To work properly,  the file must contain regular meshes in NetCDF format. The reading is made using xarray so it admits the OpenDAP protocol to open files through thredds servers.

For each input that we want to read from external files, it reads the dict used in `inputs`_ with the characteristics of the 
data passed, using the :func:`LAGAR.InputFields` module. We must copy the following block of code to prepare the
interpolant in the form F(r(t),t) for each field that we want to make an interpolant, including the mask field. ::

The different dictionaries inside the dataset correspond to the different fields to be read. Here, we are reading the flow velocity field and the mask field.

The keyword :term:`dims_dataset`, specify the dimensions of the dataset in the
order to be read. That is, the dataset will be transposed to fit in the order
chose. 

The keyword :term:`dims_fields`, specify the dimensions of each field to be read,
and :term:`dims_order`, specifies the order of the dimensions read, and change if
it is necessary. 

.. admonition:: why should I use this?

   In some datasets, the order of the dimensions is not increasing and turns into an error at the time of creating the interpolants.

Finally, the keyword, :term:`fields`, specifies the variables to be read from the file.
In case that we want to read a vector field such as in the example used below,
we should pass the :term:`fields` in a list.

::

    "input" :{
            "velocity":{
                "file_name":"MOHID_Vigo_20180723_0000.nc4",
                "dims_dataset":["lon","lat","depth","time"],
                "dims_fields":["lon","lat","depth","time"],
                "dims_order":[1,1,-1,1],
                "fields":["u","v","w"]
                },
            "mask":{
                "file_name":"MOHID_Vigo_20180723_0000.nc4",
                "dims_dataset":["lon","lat","depth","time"],
                "dims_fields":["lon","lat","depth"],
                "dims_order":[1,1,-1,1],
                "fields":["mask"]
                }
            },


Solvers
-------
This part controls the selection of the :term:`solver` in :func:`LAGAR.Solvers`.
The :term: "time_range", are the time limits to perform the integration (in the example below, it is in seconds) and :term: "dt", controls the time step.

::

    "solver":{
            "solver":"RK4",
            "time_range" : [1,120600],
            "dt": 3600
            },


Kernels
-------
This part control the selection of kernel inside the kernels module :func:`LAGAR.Kernels`.

::

    "kernel":{
            "kernel": "LagrangianSpherical2D"
            },


Outputs
-------
The ouput dictionary controls the following aspects:

::

    "output" :{
            "store_type": "Mem",
            "path_save": "./VIGO/",
            "pattern":"t_",
            "type":".nc",
            "var_names":["lon_t","lat_t","depth_t"],
            }

The :term:"store_type" has two available options, "Mem" or "Disk". In the first, case the steps or solution from solvers, can be stored into a big array in ram memory and then it is turned into a .nc or CSV file. If the computation is heave on memory and we desire to store the data on disk, we can save intermediate steps into multiple binary files. The place where these files are saved is set by the terms :term:"path_save" and :term:"pattern".

The :term:`var_names' controls the names used in the output for the saved variables in the case that we want to store it as CSV or NetCDF. In the case of, we want to store it as nc files, these names must be different from dimensions in the :term:`r0', to avoid name collision.
