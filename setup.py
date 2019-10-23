import setuptools

setuptools.setup(
    name='LAGAR',
    version='0.1',
    author='Angel Daniel Garaboa Paz, Vicente Pérez Muñuzuri',
    author_email='angeldaniel.garaboa@usc.es',
    packages=['LAGAR','examples','docs'],
    license='GPLv3',
    url='https://github.com/DanielGaraboaPaz/LAGAR',
    description='A python package for LAGrangian Analysis and Research',
    long_description=open('README.rst').read(),
    install_requires=[
              "numpy",
              "xarray",
              "scikit-image",
              "netcdf4",
              "interpolation",	
      ],
)
