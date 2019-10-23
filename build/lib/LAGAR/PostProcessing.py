import xarray as xr
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label
from numba import jit


class FTLE:
    """ 



    Attributes:
        - alias (dict): Dictionary to link the variable outputs of your lagrangian model with the standard x,y,z variables to compute the FTLE.
        - ds (xarray.dataset): Netcdf input file in xarray.dataset format.
        - ds_output (xarray.dataset): Netcdf input file in xarray.dataset format.
        - grid_coords_labels (list): Internal coords names used by MYCOASTFTLE module.
        - grid_shape (list): Dimensions of the grid used to perform the advection (optional).
        - model_input (string): Name of the Lagrangian model used. 
        - spherical (boolean): True/false if the model is in cartesian or spherical coordinates.
        - time (ds.Datarray):Datarray of 'time' aliases positions from netcdf input.
        - vars_labels (list): Private variables names used by MYCOASTFTLE module.
        - x (xarray.Datarray): Datarray of 'x' aliases positions from netcdf input.
        - x0 (xarray.Datarray): Datarray of 'x0' grid coords.  
        - y (xarray.Datarray): Datarray of 'y' aliases positions from netcdf input.
        - y0 (xarray.Datarray): Datarray of 'y0' grid coords.
        - z (xarray.Datarray): Datarray of 'z' aliases positions from netcdf input.
        - z0 (xarray.Datarray):Datarray of 'z0' grid coords.



    """

    def __init__(self, alias, spherical, model_input, grid_shape, integration_time_index):
        """
        
        Init the object with the setup provided in order to extract ftle.
        
        Args:
            - alias (dict): Dictionary to link the variable names of your lagrangian model with the private variables with the standard x,y,z variables used to compute the FTLE.
            - spherical (boolen): True/false if the model is in cartesian or spherical coordinates.
            - model_input (string): Name of the Lagrangian model used. 
            - grid_shape (list): Dimensions of the grid of initial conditions used to perform the advection (optional)

            Inputs supported:
                - Pylag
                - LAGAR

        Example:
            - pylag_ouput.nc
                output_variables: xpos, ypos, zpos
            -alias_dict:
                alis_dictionary={'x':'xpos','y':'ypos','z':'zpos'}   

        Deleted Parameters:
            **output_dict: Description

        """

        self.model_input = model_input

        # Here we stablish the correspondece between the Lagrangian model input and also
        # the private variables x,y,z of the MYCOASTFTLE module.
        # Each model produces the outputs with their internal variable names.
        # The alias dictionary "links" those variables with the
        # internal x,y,z of the MYCOAST_FTLE module.
        # There are Two defaults dictionaries for LAGAR and PYLAG.

        if self.model_input == 'pylag':
            self.alias = {'x': 'xpos', 'y': 'ypos',
                          'z': 'zpos', 'time': 'time'}
        elif self.model_input == 'lagar':
            self.alias = {'x': 'lon', 'y': 'lat', 'z': 'depth', 'time': 'time',
                          'x0': 'x_0', 'y0': 'y_0', 'z0': 'z_0'}
        else:
            self.alias = alias

        self.spherical = spherical

        # grid_shape variable is used for those Lagrangian models which provided the
        # the evolution of particle trajectories in a array of the form. x[time,particles_id],
        # y[time,particles_id]. Other models produce ouputs in the format x[time,x_0,y_0]

        self.grid_shape = grid_shape

        # the var_labels and and grid_coords_labels, are the internal (or private) variables
        # used to compute the FTLE and LCS, in 2D or 3D dimensions.

        self.vars_labels = ['x', 'y']
        self.grid_coords_labels = ['time', 'y0', 'x0']
        if 'z' in self.alias:
            self.vars_labels = ['x', 'y', 'z']
            self.grid_coords_labels = ['time', 'z0', 'y0', 'x0']
        self.ds_output = []

        # time index on lagrangian used to compute the FTLE
        self.integration_time = integration_time_index


    def get_input(self, netcdf_file):
        self.ds = xr.open_dataset(netcdf_file).fillna(0)
        
        print('*****************************')
        print('Processing file:',netcdf_file)
        print('*****************************')

    def set_alias(self):
        """
        This method set the alias names in case that you use you own dictionary 
        for your own Lagrangian model.

        Args:
            

        Example:
            - YourLagrangianModel.nc
                output_variables: longitude_t, latitude_t, depth_t, t
            - alias_dict:
                alias_dictionary={'x':'longitude_t','y':latitude_t','depth':'z_t','time':'t'}   

        """
        
        
        print('Link variables of model input: '+ self.model_input)
        print(self.alias,'\n')

        if self.alias['x']:
            self.x = self.ds[self.alias['x']]
        if self.alias['y']:
            self.y = self.ds[self.alias['y']]
        if self.alias['z']:
            self.z = self.ds[self.alias['z']]
        if self.alias['time']:
            self.time = self.ds[self.alias['time']]

        # If the dataset has the Lagrangian trajectories in variables an array grid_shape format
        # that is (x[time,x_0,y_0]) it is suposed that the grid_shape is included in the netcdf dimensions.
        # and the dictionary should link the netcdf initial conditions (coordinates) with the private coordinates
        # x0,y0,y0

        # Example:

        if len(self.grid_shape) == 0:

            self.x0 = self.ds[self.alias['x0']]
            self.y0 = self.ds[self.alias['y0']]
            self.z0 = self.ds[self.alias['z0']]

    def transform_input(self):
        """

        This functions turns the input dataset comming from lagrangian model
        input into a compatible structured dataset to compute FTLE.
        New models will be available in the future.


        Args:
            - self(MYCOAST_FTLE) : MYCOAST_FTLE instance

        Returns:
            - self(MYCOAST_FTLE): Return a MYCOAST_FTLE instance ready for FTLE computation.

        """


        if self.model_input == 'pylag':
            
            print('Turning '+ self.model_input +' input into a structured grid netcdf')
            print('from',self.x.isel(time=0).shape, 'to', self.grid_shape,'\n')

            if self.z.size > 0:
                nvars = 3
                # In case that variables are stored in a array format. We use the
                # use the first time instant (timeindex=0) to create the grids coordinates
                # with the sizes of grid_shape


                self.x0 = self.x.isel(time=0).values.reshape(
                    self.grid_shape)[0, 0, :]
                self.y0 = self.y.isel(time=0).values.reshape(
                    self.grid_shape)[0, :, 0]
                self.z0 = self.z.isel(time=0).values.reshape(
                    self.grid_shape)[:, 0, 0]
                
                # Setting the grid coordinates
                coords = dict(zip(self.grid_coords_labels, zip(
                    self.grid_coords_labels, [self.time, self.z0, self.y0, self.x0])))

                # Transform the variables into a structure grid
                variables_in_grid_form = [var_to_reshape.values.reshape(
                    [self.time.size]+self.grid_shape) for var_to_reshape in [self.x, self.y, self.z]]
                
                #Transform the variables into a dataset format.
                variables_in_ds_form = dict(zip(self.vars_labels, zip(
                    nvars*[self.grid_coords_labels], list(variables_in_grid_form))))

                #
                self.ds_output = xr.Dataset(
                    variables_in_ds_form, coords=coords)

                self.x = self.ds_output.x
                self.y = self.ds_output.y
                self.z = self.ds_output.z

            else:
                
                

                nvars = 2
                self.x0 = self.x.isel(time=0).values.reshape(
                    self.grid_shape)[0, :]
                self.y0 = self.y.isel(time=0).values.reshape(
                    self.grid_shape)[:, 0]

                coords = dict(zip(self.grid_coords_labels, zip(
                    self.grid_coords_labels, [self.time, self.y0, self.x0])))
                variables_in_grid_form = [var_to_reshape.values.reshape(
                    self.grid_shape + [self.time.size]) for var_to_reshape in [self.x, self.y]]
                variables_in_ds_form = dict(zip(self.vars_labels, zip(
                    nvars*[self.grid_coords_labels], variables_in_grid_form)))

                self.ds_output = xr.Dataset(
                    variables_in_ds_form, coords=coords)
                self.x = self.ds_output.x
                self.y = self.ds_output.y

        elif self.model_input == 'lagar':
            
             print('Turning '+ self.model_input +' input into a structured grid netcdf')
             
             if self.z.size > 0:
                nvars = 3
                coords = dict(zip(self.grid_coords_labels, zip(
                    self.grid_coords_labels, [self.time, self.z0, self.y0, self.x0])))
                variables_in_ds_form = dict(zip(self.vars_labels, zip(
                    nvars*[self.grid_coords_labels], [self.x, self.y, self.z])))
                self.ds_output = xr.Dataset(
                    variables_in_ds_form, coords=coords)
                self.x = self.ds_output.x
                self.y = self.ds_output.y
                self.z = self.ds_output.z

             else:
                nvars = 2
                coords = dict(zip(self.grid_coords_labels, zip(
                    self.grid_coords_labels, [self.time, self.y0, self.x0])))
                variables_in_ds_form = dict(zip(self.vars_labels, zip(
                    nvars*[self.grid_coords_labels], [self.x, self.y])))
                self.ds_output = xr.Dataset(
                    variables_in_ds_form, coords=coords)
                self.x = self.ds_output.x
                self.y = self.ds_output.y
        
        else:
            self.ds_output = self.ds
            print(f"\033[7;1;5;{'31m'}{'WARNING:You are redirecting the input to the output.:WARNING'}\033[00m")
            print(f"\033[7;1;5;{'31m'}{'Your input must be written with a structured grid format:WARNING'}\033[00m")


        return

    def get_ftle(self, to_dataset=True):
        """

        It computes the FTLE (Finite Time Lyapunov Exponents) using the 
        Cauchy Green finite time deformation tensor described, Shadden (2005).
        

        Args: 
            - T (float, optional): If the dataset has no time attribute in datetimeformat,
            you can provide the advection time.
            - timeindex (int, optional): Index time to select the to compute the FTLE.
            By default is the last time step of the Lagrangian advection.
            - to_dataset (bool, optional): By default, it added the computed FTLE field,
            to the output dataset

        Returns:
            self: Added the FTLE_forward or FTLE_backward field to the ds_output.

        """

        if type(self.integration_time) == int:
            timeindex = self.integration_time
            T = (self.time[timeindex]-self.time[0]
                 ).values.astype('timedelta64[s]').astype('f8')
            if T == 0.0:
                T = (self.time[timeindex]-self.time[0]
                 ).values.astype('f8')*1e-9
        elif type(self.integration_time) == float:
            timeindex = -1
            T = self.integration_time

        print('Getting FTLE for a integration time of',T, 'seconds \n')
        
        flag_3d = hasattr(self, 'z0') & (self.z0.size > 1)

        if (self.spherical == False) and (flag_3d == False):

            dxdy, dxdx = np.gradient(self.x.isel(
                time=timeindex).squeeze(), self.y0, self.x0)
            dydy, dydx = np.gradient(self.y.isel(
                time=timeindex).squeeze(), self.y0, self.x0)

            ny, nx = dxdy.shape
            ftle = np.zeros([ny, nx])
            J = np.zeros([2, 2])

            for i in range(0, ny):
                for j in range(0, nx):
                    J = np.array([[dxdx[i, j], dxdy[i, j]],
                                  [dydx[i, j], dydy[i, j]]])
                    C = np.dot(np.transpose(J), J)
                    eigLya, _ = np.linalg.eigh(C)
                    ftle[i, j] = (1./T)*np.log(np.sqrt(eigLya.max()))

        elif (self.spherical == True) and (flag_3d == False):

            R = 6370000.

            dxdy, dxdx = np.gradient(self.x.isel(
                time=timeindex).squeeze(), self.y0, self.x0)
            dydy, dydx = np.gradient(self.y.isel(
                time=timeindex).squeeze(), self.y0, self.x0)

            ny, nx = np.shape(dxdx)
            ftle  = np.zeros([ny, nx])

            theta = self.y.isel(time=timeindex).squeeze().values
            for i in range(0, ny):
                for j in range(0, nx):
                    J = np.array([[dxdx[i, j], dxdy[i, j]],
                                  [dydx[i, j], dydy[i, j]]])
                    M = np.array(
                        [[R*R*np.cos(theta[i, j]*np.pi/180.), 0], [0., R*R]])
                    C = np.dot(np.dot(np.transpose(J), M), J)
                    eigLya, _ = np.linalg.eigh(C)
                    ftle[i, j] = (1./T)*np.log(np.sqrt(eigLya.max()))

        elif (self.spherical == False) and (flag_3d == True):

            dxdz, dxdy, dxdx = np.gradient(self.x.isel(
                time=timeindex).values.squeeze(), self.z0, self.y0, self.x0)
            dydz, dydy, dydx = np.gradient(self.y.isel(
                time=timeindex).values.squeeze(), self.z0, self.y0, self.x0)
            dzdx, dzdy, dzdz = np.gradient(self.z.isel(
                time=timeindex).values.squeeze(), self.z0, self.y0, self.x0)

            nz, ny, nx = dxdz.shape
            ftle = np.zeros([nz, ny, nx])

            J = np.zeros([3, 3])

            for i in range(0, nz):
                for j in range(0, ny):
                    for k in range(0, nx):
                        J = np.array([[dxdx[i, j, k], dxdy[i, j, k], dxdz[i, j, k]],
                                      [dydx[i, j, k], dydy[i, j, k], dydz[i, j, k]],
                                      [dzdx[i, j, k], dzdy[i, j, k], dzdz[i, j, k]]])
                        C = np.dot(np.transpose(J), J)
                        eigLya, _ = np.linalg.eig(C)
                        ftle[i, j, k] = (1./T)*np.log(np.sqrt(eigLya.max()))

        elif((self.spherical == True) and (flag_3d == True)):
            print('No spherical 3D FTLE available at the moment')

        if to_dataset == True:
            if T>0:
                self.ds_output['FTLE_forward'] = (
                self.x.isel(time=timeindex).dims, ftle)
            if T<0:
                self.ds_output['FTLE_backward'] = (
                self.x.isel(time=timeindex).dims, ftle)

        return ftle
            

    def explore_ftle_timescale(self):
        """

        It computes the FTLE for all timesteps instead of a given one.
        The output produced will help you  to explore  the timescale of the deformation 
        in order to infer the attributes for LCS and FTLE computation.


        Args:
            - to_dataset (bool, optional): By default, it added the computed FTLE field,
            to the output dataset.
 

        Returns:
            self(MYCOASTFTLE): ds_output with FTLE computed for all timesteps.

        """

        ftle = np.zeros_like(self.ds_output.x.values)
        nsteps = self.ds_output.time.size

        for i in range(0, nsteps):
            self.integration_time = i
            ftle[i] = self.get_ftle(to_dataset=False)
            print('Computing FTLE field for step'+str(i)+'Percentage:'+str(100.*i/nsteps)+'\n')
        self.ds_output['FTLE'] = (self.x.dims, ftle)
        return 

            
        
class LCS:

    def __init__(self, lag_ftle, eval_thrsh='infer', ftle_thrsh='infer', area_thrsh=100, nr_neighb=8, ridge_points=False, to_dataset=True):

        self.lag_ftle = lag_ftle
        self.eval_thrsh = eval_thrsh
        self.ftle_thrsh = ftle_thrsh
        self.area_thrsh = area_thrsh
        self.nr_neighb = nr_neighb
        self.ridge_points = ridge_points

    def get_lcs(self, to_dataset=True):
        """


         Extract points that sit on the dominant ridges of FTLE 2D data
         A ridge point is defined as a point, where the gradient vector is orthogonal
         to the eigenvector of the smallest (negative) eigenvalue of the Hessian
         matriv (curvature).
         The found points are filtered to extract only points on strong FTLE
         separatrices. Therefore points are only taken, if:

         a) FTLE value is high at the point's position
         b) the negative curvature is high
       
        
        Args:
            - eval_thrsh (str or float, optional): scalar. Selects zones with small Hessian eigenvalue smaller than eval_thrsh. use 'infer' to obtain the threshold from the 95 percentile of the data of the FTLE field.
            - FTLE_thrsh (str or float, optional): scalar. Selects connected areas (with 4 or 8 neighbors) larger than area_thrsh. use 'infer' to obtain the threshold from the 95 percentile of the data of the FTLE field.
            - area_thrsh (float, optional):  scalar. Selects connected areas (with 4 or 8 neighbors) larger than area_thrsh
            - nr_neighb (int, optional): scalar. Connection neighbours (4 or 8)
            - ridge_points (bool, optional): x0,y0 exact ridge poisition if 1. (matrix coordinates)  
            - to_dataset (bool, optional): Logical mask for ridges in the FTLE field LCS_forward and LCS_backward to the outputted dataset.

            
        Example:
             Define variables
             eval_thrsh = -0.005;        # Selects zones with small Hessian eigenvalue smaller than eval_thrsh
             FTLE_thrsh = 0.07;          # Selects zones with FTLE larger than FTLE_thrsh
             area_thrsh = 10;            # Selects connected areas (with 4 or 8 neighbors) larger than area_thrsh
             nr_neighb = 8;              # conection neighbours (4 or 8)

        Returns:
            combined_bin: logical mask for ridges in FTLE field
            x0: Positions of points on FTLE ridges
            y0: Positions of points on FTLE ridges
        """

        
        if 'FTLE_forward' in self.lag_ftle.ds_output.keys():
            ftle = self.lag_ftle.ds_output['FTLE_forward'].fillna(0).squeeze().values
            m, n = ftle.shape
        elif 'FTLE_backward' in self.lag_ftle.ds_output.keys():
            ftle = self.lag_ftle.ds_output['FTLE_backward'].fillna(0).squeeze().values
            m, n = ftle.shape

        if self.ftle_thrsh == 'infer':
            self.ftle_thrsh = np.percentile(ftle, 95)

        # Gradient and Hessian matrix (2nd derivatives) from finite differences
        [dy, dx] = np.gradient(ftle)

        # Eigenvalues of Hessian matrix (analytically)
        # EVal = min( 1/2*(a+d) + sqrt(b.*c + 4*(a-d).^2), 1/2*(a+d) - sqrt(b.*c + 4*(a-d).^2) );
        # psu = (a-EVal-c)./(d-EVal-b);
        # Smaller (negative) Eigenvector of Hessian matrix (analytically)
        # EVecx = 1./sqrt(1^2 + psu.^2);
        # EVecy = psu./sqrt(1^2 + psu.^2);

        # Make 2D hessian
        hxx, hxy, hyy = hessian_matrix(ftle, sigma=3)

        i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)

        Lambda2 = np.zeros_like(hxx)
        Lambda1 = np.zeros_like(hxx)
        EVecx = np.zeros_like(hxx)
        EVecy = np.zeros_like(hxx)

        for i in range(0, m):
            for j in range(0, n):
                eigvalue, eigenvec = np.linalg.eigh(
                    np.matrix([[hxx[i, j], hxy[i, j]], [hxy[i, j], hyy[i, j]]]))
                Lambda2[i, j], Lambda1[i, j], = eigvalue
                EVecx[i, j], EVecy[i, j] = eigenvec[1, 0], eigenvec[1, 1]

        EVal = np.minimum(Lambda2, Lambda1)
        EVal = np.nan_to_num(EVal)

        # Compute the direction of the minor eigenvector
        # Define ridges as zero level lines of inner product 
        # of gradient and eigenvector of smaller (negative) eigenvalue
        ridge = EVecx*dx + EVecy*dy

        
        if self.eval_thrsh == 'infer':
            self.eval_thrsh = np.percentile(EVal, 95)

        # Define filter masks with high negative curvature
        Eval_neg_bin = EVal < self.eval_thrsh

        # High FTLE value
        FTLE_high_bin = ftle > self.ftle_thrsh

        # Combined mask
        combined_bin = FTLE_high_bin*Eval_neg_bin

        # Remove small connected areas in combined mask

        # label areas
        L = label(combined_bin, neighbors=self.nr_neighb)
        # filter mask
        Lmax = L.max()
        for i in range(1, Lmax):
            if ((L[L == i]).size < self.area_thrsh):  # || (Lmax_ind > L_f/2)
                L[L == i] = 0

        # set new mask
        combined_bin[L > 0] = 1
        combined_bin[L == 0] = 0

        print('Number of potential LCS founded:', Lmax)
        print('Area threshold:', self.area_thrsh)
        print('FTLE threshold:', self.ftle_thrsh)
        print('EVal threshold:', self.eval_thrsh)

        if to_dataset == True:
            if 'FTLE_forward' in self.lag_ftle.ds_output.keys():
                self.lag_ftle.ds_output['LCS_forward'] = (
                    self.lag_ftle.ds_output['FTLE_forward'].dims, combined_bin[np.newaxis])
            elif 'FTLE_backward' in self.ds_output.keys():
                self.lag_ftle.ds_output['LCS_backward'] = (
                    self.lag_ftle.ds_output['FTLE_backward'].dims, combined_bin[np.newaxis])

        # DEFINE RIDGE POINTS
        # search ridge==0 contour lines
        # suche Nulldurchgang fuer Gerade zwischen zwei direkt benachbarten grid Punkten
        # setze den Punkt, wenn der Nulldurchgang zwischen den beiden grid Punkten
        # liegt
        # clear x0 y0 x1 y1 x y
        if self.ridge_points == False:
            return combined_bin

        if self.ridge_points == True:
            x0 = 0
            y0 = 0

            m, n = np.shape(FTLE)
            X = np.arange(1, n)
            Y = np.arange(1, m)
            cnt = 0
            x0 = []
            y0 = []
            for i in np.arange(0, m-1):
                for j in np.arange(0, n-2):
                    if combined_bin[i, j] == 1:
                        x = X[j:j+2]
                        z = np.array([ridge[i, j], ridge[i, j+1]])
                        m1 = (z[1]-z[0])/(x[1]-x[0])  # Calculamos la pendiente
                        # Calculamos el punto de corte con el eje y,
                        n1 = z[0] - m1*x[0]
                        # round zero point (-n1/m1) to the nearest j
                        if(np.abs(-n1/m1-x[0]) < np.abs(-n1/m1-x[1])):
                            js = j
                        else:
                            js = j+1
                        if(x[0] < -n1/m1 and -n1/m1 < x[1] and combined_bin[i, js] == 1):
                            cnt = cnt+1
                            x0.append(-n1/m1)
                            y0.append(Y[i])

            for i in np.arange(0, m-2):
                for j in np.arange(0, n-1):
                    if combined_bin[i, j] == 1:
                        y = Y[i:i+2]
                        z = np.array([ridge[i, j], ridge[i+1, j]])
                        m1 = (z[1]-z[0])/(y[1]-y[0])
                        n1 = z[0] - m1*y[0]
                        # round zero point (-n1/m1) to the nearest i
                        if(np.abs(-n1/m1-y[0]) < np.abs(-n1/m1-y[1])):
                            isi = i
                        else:
                            isi = i+1

                        if(y[0] < -n1/m1 and -n1/m1 < y[1] and combined_bin[isi, j] == 1):
                            cnt = cnt+1
                            x0.append(X[j])
                            y0.append(-n1/m1)

                    return x0, y0, combined_bin



class GridMeasures:
    
    def __init__(self, alias, model_input, bins_option, nbins, integration_time_index):
        
        
        """
        
        Init the object with the setup provided in order to extract ftle.
        
        Args:
            - alias (dict): Dictionary to link the variable names of your lagrangian model with the private variables with the standard x,y,z variables used to compute the FTLE.
            - spherical (boolen): True/false if the model is in cartesian or spherical coordinates.
            - model_input (string): Name of the Lagrangian model used. 
            - grid_shape (list): Dimensions of the grid of initial conditions used to perform the advection (optional)
        
            Inputs supported:
                - Pylag
                - LAGAR
        
        Example:
            - pylag_ouput.nc
                output_variables: xpos, ypos, zpos
            -alias_dict:
                alis_dictionary={'x':'xpos','y':'ypos','z':'zpos'}   
        
        Deleted Parameters:
            **output_dict: Description
        
        """
        
        self.nbins = nbins
        self.bins_option = bins_option
        self.ds = []
        self.ds_output = []
        self.da_output = {} 
        self.model_input = model_input
        self.integration_time_index = integration_time_index
        self.grid_shape = []
        
        # Here we stablish the correspondece between the Lagrangian model input and also
        # the private variables x,y,z of the MYCOASTFTLE module.
        # Each model produces the outputs with their internal variable names.
        # The alias dictionary "links" those variables with the
        # internal x,y,z of the MYCOAST_FTLE module.
        # There are Two defaults dictionaries for LAGAR and PYLAG.
        if self.nbins:
            self.nbis = nbins
        else:
            self.nbins = 100
                
        if self.model_input == 'pylag':
            self.alias = {'x': 'xpos', 'y': 'ypos',
                          'z': 'zpos', 'time': 'time'}
        elif self.model_input == 'lagar':
            self.alias = {'x': 'lon', 'y': 'lat', 'z': 'depth', 'time': 'time',
                          'x0': 'x_0', 'y0': 'y_0', 'z0': 'z_0'}
        else:
            self.alias = alias
        
    def get_input(self, netcdf_file):
        self.ds = xr.open_dataset(netcdf_file).fillna(0)
        
        print('*****************************')
        print('Processing file:',netcdf_file)
        print('*****************************')
        
    
    def set_alias(self):
        """
        This method set the alias names in case that you use you own dictionary 
        for your own Lagrangian model.

        Args:
            

        Example:
            - YourLagrangianModel.nc
                output_variables: longitude_t, latitude_t, depth_t, t
            - alias_dict:
                alias_dictionary={'x':'longitude_t','y':latitude_t','depth':'z_t','time':'t'}   

        """
        
        print('*****************************')
        print('Link variables of model input: '+ self.model_input)
        print(self.alias,'\n')
        print('*****************************')

        if 'x' in self.alias:
            self.x = self.ds[self.alias['x']]
        if 'y' in self.alias:
            self.y = self.ds[self.alias['y']]
        if 'z' in self.alias:
            self.z = self.ds[self.alias['z']]
        if 'time' in self.alias:
            self.time = self.ds[self.alias['time']]

        # If the dataset has the Lagrangian trajectories in variables an array grid_shape format
        # that is (x[time,x_0,y_0]) it is suposed that the grid_shape is included in the netcdf dimensions.
        # and the dictionary should link the netcdf initial conditions (coordinates) with the private coordinates
        # x0,y0,y0

        # Example:


        if 'x0' in self.alias:
            self.x0 = self.ds[self.alias['x0']]
            
        if 'y0' in self.alias:
            self.y0 = self.ds[self.alias['y0']]
         
        if 'z0' in self.alias:
            self.z0 = self.ds[self.alias['z0']]

            
        

    def get_residence_time(self, to_dataset=True):
        """

        It computes the raw residence time. 
        This function counts the timesteps that a particle spetn in a box.
        It doesn't matter if it is just a particle or a bunch of them. 


        Args:
            -timeindex (TYPE, optional): Description
            bins_option (str, optional): Description
            nbins (int, optional): Description
            to_dataset (bool, optional): Description

        Returns:
            TYPE: Description


        """
        if self.bins_option == 'domain':
            x_bins = np.linspace(self.x.min(), self.x.max(), self.nbins)
            y_bins = np.linspace(self.y.min(), self.y.max(), self.nbins)
            if hasattr(self,'z'):
                z_bins = np.linspace(self.z.min(), self.z.max(), self.nbins)
                
        elif self.bins_option == 'origin':
            if self.nbins:
                x_bins = np.linspace(self.x0.values.min(),self.x0.values.max(),self.nbins)
                y_bins = np.linspace(self.y0.values.min(),self.y0.values.max(),self.nbins)
                if hasattr(self,'z'):
                    z_bins = np.linspace(self.z0.values.min(),self.z0.values.max(),self.nbins)
            else:
                 x_bins = self.x0.values
                 y_bins = self.y0.values
                 if hasattr(self,'z'):
                     z_bins = self.z0.values
                     
        elif self.bins_option == 'personalized':
            x_bins = np.linspace(*self.nbins[0])
            y_bins = np.linspace(*self.nbins[1])
            if hasattr(self,'z'):
                z_bins = np.linspace(*self.nbins[2])

        print('**********************************')
        print('Computing residence time in domain:')
        print('**********************************')
        print('x: ',x_bins.min(),' ',x_bins.max())
        print('y: ',y_bins.min(),' ',y_bins.max())
        if hasattr(self,'z'):
             print('z: ',z_bins.min(),' ',z_bins.max())
        print('nbins:',self.nbins)
        print('dt:', (self.time[1]-self.time[0]).values.astype('f8')*1e-9)
       

        bins = (y_bins, x_bins)

        x_centers = (x_bins[:-1] + x_bins[1:]) / 2.
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2.
        
        if hasattr(self,'z'):
             z_centers = (z_bins[:-1] + z_bins[1:]) / 2.
             bins = (z_bins, y_bins, x_bins)

        
        H_integrated = np.zeros((y_centers.size, x_centers.size))
        
        if hasattr(self,'z'):
            H_integrated = np.zeros((z_centers.size, y_centers.size, x_centers.size))

        if hasattr(self,'z'):
            for i in range(1, self.time.size):
                data_to_histogram = np.stack((self.z.isel(time=i).values.ravel(),\
                                              self.y.isel(time=i).values.ravel(),\
                                              self.x.isel(time=i).values.ravel()),axis=1)
                
                H,_ = np.histogramdd(data_to_histogram, bins = bins)
                
                dt = (self.time[i]-self.time[i-1]).values.astype('f8')*1e-9
    
                H[H > 0] = 1.
    
                H_integrated = H*dt + H_integrated
            
            coords = {'z_c': ('z_c', z_centers), 'y_c': ('y_c', y_centers), 'x_c': ('x_c', x_centers)}
            dims = ['z_c', 'y_c', 'x_c']
        
            if to_dataset == True:
                self.da_output['residence_time'] = xr.DataArray(H_integrated, coords=coords, dims=dims)
            else:
                return H_integrated
                
        else:   
            for i in range(1, self.time.size):
                H, _, _ = np.histogram2d(self.y.isel(time=i).values.ravel(),\
                                         self.x.isel(time=i).values.ravel(),\
                                         bins = bins)
                # turn the data from nanoseconds by default to seconds using 1e-9
                # Is f we use datetime to seconds, it round the seconds till 0.
                dt = (self.time[i]-self.time[i-1]).values.astype('f8')*1e-9
    
                H[H > 0] = 1.
    
                H_integrated = H * dt + H_integrated

        # neglect extreme values
        #H_integrated[(H_integrated == 0) | (
            #H_integrated == np.max(H_integrated))] = np.nan
    
            coords = {'y_c': ('y_c', y_centers), 'x_c': ('x_c', x_centers)}
            dims = ['y_c', 'x_c']
        
            if to_dataset == True:
                self.da_output['residence_time']= xr.DataArray(H_integrated, coords=coords, dims=dims)
            else:
                return H_integrated

    def get_concentrations(self, to_dataset=True):
        """

        It computes the raw residence time. 
        This function counts the timesteps that a particle spetn in a box.
        It doesn't matter if it is just a particle or a bunch of them. 


        Args:
            timeindex (TYPE, optional): Description
            bins_option (str, optional): Description
            nbins (int, optional): Description
            to_dataset (bool, optional): Description

        Returns:
            TYPE: Description


        """
        if self.bins_option == 'domain':
            x_bins = np.linspace(self.x.min(), self.x.max(), self.nbins)
            y_bins = np.linspace(self.y.min(), self.y.max(), self.nbins)
        elif self.bins_option == 'origin':
            if self.nbins:
                x_bins = np.linspace(self.x0.values.min(),self.x0.values.max(),self.nbins)
                y_bins = np.linspace(self.y0.values.min(),self.y0.values.max(),self.nbins)
            else:
                 x_bins = self.x0.values
                 y_bins = self.y0.values
        elif self.bins_option == 'personalized':
            x_bins = np.linspace(*self.nbins[0])
            y_bins = np.linspace(*self.nbins[1])
                 
       
        bins = (x_bins, y_bins)

        x_centers = (x_bins[:-1] + x_bins[1:]) / 2.
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2.

        print('**********************************')
        print('Computing concentraions in domain:')
        print('**********************************')
        print('x :',x_bins.min(),' ',x_bins.max())
        print('y :',y_bins.min(),' ',y_bins.max())
        print('nbins:',self.nbins)
        
        
        
        concentrations = np.zeros((self.ds.time.size, y_centers.size, x_centers.size))

        for i in range(0, self.time.size):
            concent, _, _ = np.histogram2d(self.x.isel(time=i).values.ravel(), self.y.isel(time=i).values.ravel(), bins=bins)
            concentrations[i, :, :] = concent.transpose()

        coords = {'time': self.time, 'y_c': (
            'y_c', y_centers), 'x_c': ('x_c', x_centers)}
        dims = ['time', 'y_c', 'x_c']
        
        if to_dataset == True:
            self.da_output['concentrations'] = xr.DataArray(concentrations, coords, dims=dims)
        else:
            return concentrations
        
    
    def get_netcdf(self,file_output):
        self.ds_output = xr.Dataset(self.da_output)
        self.ds_output.to_netcdf(file_output)



#def eddy_analysis_2d(u,v,lat,lon):
#     """
#     Function under construction.
#     This function computes different Eulerian quantities related to vorticity and eddys.
    
#     Args:
#         u (np.array): Longitudinal component of the velocity field in the form [time,lat,lon].

#         v (np.array): Latitudinal component of the velocity field in the form [time,lat,lon].

#         lat (np.array): Latitude array.

#         lon (np.array): Longitude array.
    
#     Returns:
#         TYPE: Description
#     """

#     R_T = 6370000.
#     lon_mesh,lat_mesh = np.meshgrid(lon,lat)
#     dlat = R_T*(np.pi/180.)*(lat[1]-lat[0])
#     dlon = R_T*(np.pi/180.)*(lon[1]-lon[0])*np.cos((np.pi/180.)*lat_mesh)
#     darea = dlat*dlon
    

#     OW = np.zeros_like(u)
#     Curl = np.zeros_like(u)
#     Div = np.zeros_like(u)
#     Eddy = np.zeros_like(u)
    
#     for t in range(0,u.shape[0]):
        
#         dudy,dudx = np.gradient(u[i,:,:], dlat,dlon)
#         dvdy,dvdx = np.gradient(v[i,:,:], dlat,dlon)
    
#         w  = dvdx - dudy
#         sn = dudx - dvdy
#         ss = dvdx + dudy
        
#         OW[i,:,:] = np.power(sn,2) + np.power(ss,2) - np.power(w,2)
#         Curl[i,:,:] = w
#         Div[i,:,:] = dudx + dvdy
        
#         Eddy[i,:,:] = (OW[t,:,:] < -np.std(OW[t,:,:])) 
        
#     Labels =  np.expand_dims(measure.label(Eddy[:,:,:]),axis=1) 

#     if (AreaEnergy == True):
#         Area = np.zeros_like(u)
#         Energy = np.zeros_like(u)
#         for i in range(0,ds.time.size):
#             #print ds.time.size,i
#             for j in range(Labels[i,0,Labels[i,0,:,:]>0].min(),Labels[i,0,Labels[i,0,:,:]>0].max()):
#                 Mask = (Labels[i,0,:,:] == j)
#                 Area[i,0,Mask] = darea[Mask].sum()  
#                 Energy[i,0,Mask] = np.nansum(Mask*darea*0.5*(np.power(ds.u[i,0,:,:],2)*+np.power(ds.v[i,0,:,:],2)))/ np.sum((darea[Mask]))   
    
    
#     return