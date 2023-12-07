from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


gcc_flags = [] #['-shared', '-O2', '-fno-strict-aliasing']

ext_special = Extension('functions_special', sources=['functions_special.pyx'],
                language='c',
                #extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'],
                #extra_compile_args=['/openmp'], extra_link_args=['/openmp'],
                include_dirs = get_numpy_include_dirs())

ext_generic = Extension('functions_generic', sources=['functions_generic.pyx'],
                language='c',
                #extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'],
                #extra_compile_args=['/openmp'], extra_link_args=['/openmp'],
                include_dirs = get_numpy_include_dirs())

ext_convergence = Extension('functions_convergence', sources=['functions_convergence.pyx'],
                language='c',
                extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'],
                #extra_compile_args=['/openmp'], extra_link_args=['/openmp'],
                include_dirs = get_numpy_include_dirs())
                
ext_sampling = Extension('functions_sampling',
                sources=['c/c_functions_sampling.c','functions_sampling.pyx'],
                language='c',
                extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'],
                #extra_compile_args=['/openmp'], extra_link_args=['/openmp'],
                include_dirs = get_numpy_include_dirs())

ext_special.cython_directives = {'language_level': "3"}
ext_generic.cython_directives = {'language_level': "3"}
ext_convergence.cython_directives = {'language_level': "3"} 
ext_sampling.cython_directives = {'language_level': "3"} 

setup(name='FUNCTION MODULES',
      ext_modules = [ext_special,ext_generic,ext_convergence,ext_sampling],
      cmdclass = {'build_ext': build_ext},
      # since the package has c code, the egg cannot be zipped
      zip_safe=False)

