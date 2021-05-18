from setuptools import find_packages
from numpy.distutils.core import Extension

ext1 = Extension(name = 'singlet_yield',
                 sources = ['src/symmetric.f90', 'src/asymmetric.f90', 'src/floquet.f90', 'src/gamma_compute.f90', 'src/trajectories.f90'],
                 extra_f90_compile_args = ['-fopenmp'],
                 #extra_f90_compile_args = ['-fbounds-check'],
                 extra_link_args = ['-lgomp'],
                 f2py_options = ['--quiet'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'singlet_yield',
          package_dir = {"": "spin_dynamics"},
          ext_modules = [ext1],
          script_args = ['build_ext'],
          options = {'build_ext':{'inplace':True}}
          )
# End of setup_example.py
