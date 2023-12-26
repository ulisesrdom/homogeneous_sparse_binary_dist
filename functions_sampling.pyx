import numpy as np
import functions_special as f_sp
from libc.stdio cimport printf
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs
from libc.stdlib cimport rand, RAND_MAX

cimport numpy as np
cimport cython


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the polylogarithmic exponential distribution function.
# Parameters:
# ---F_vals   : float array where each of the N_points values of the discretized probability
#               distribution function will be stored.
# ---N_points : integer value with the number of discrete points to evaluate.
# ---delta_r  : float value with the infinitesimal dr approximation inside the integral.
# ---f        : float value with the sparsity inducing parameter.
# ---m        : integer value with the positive integer order of the polylogarithmic function.
# ---M        : integer value with the number of terms at which to truncate the infinite series
#               for evaluation of the polylogarithmic case m>1.
# Returns:
# ---No return value. The probability distribution function values are stored in the F_vals array.
@boundscheck(False)
@wraparound(False)
def F_polylogarithmic_eval( np.ndarray[float,ndim=1,mode="c"] F_vals, \
                            int N_points, float delta_r, float f, int m, int M ):
   cdef:
      int i,j
      float u,v1,Li_m_ri
   if m == 1 :
      # Exact analytical form for m = 1
      # -------------------------------------------------------------------
      if f == 1. :
         for i in range(0,N_points):
            u           = (float( i ) + 0.5) * delta_r
            F_vals[ i ] = log( 1.0 + u ) / log( 2.0 )
      else:
         for i in range(0,N_points):
            u           = (float( i ) + 0.5) * delta_r
            v1          = pow( 1.0 , -f+1.0 )
            F_vals[ i ] = (v1 - pow( 1.0+u , -f+1.0 )) / ( v1 - pow( 2.0 , -f+1.0 ) )
   else:
      # Numerical approximation for m > 1
      # -------------------------------------------------------------------
      r_disc   = np.zeros((N_points,),dtype=np.float32)
      Li_m_r   = np.zeros((N_points,),dtype=np.float32)
      for i in range(0,N_points):
         r_disc[ i ] = (float(i) + 0.5) * delta_r
         Li_m_ri     = 0.
         for j in range(1,N_points):
            Li_m_ri  = Li_m_ri + ( pow(-1.,j+1) * (1.0 / pow(j,m)) * pow(r_disc[i],j) )
         Li_m_r[ i ] = Li_m_ri
      Z        = 0.
      for i in range(0,N_points):
         Z     = Z + (exp( -f * Li_m_r[i] ) * delta_r)
      
      for i in range(0,N_points):
         u     = (float( i ) + 0.5) * delta_r
         I     = 0.
         for j in range(0,N_points):
            if r_disc[ j ] <= u :
               I  = I + (exp( -f * Li_m_r[j] ) * delta_r)
            else :
               break
         F_vals[ i ] = I / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the shifted-geometric exponential distribution function.
# Parameters:
# ---F_vals   : float array where each of the N_points values of the discretized probability
#               distribution function will be stored.
# ---N_points : integer value with the number of discrete points to evaluate.
# ---delta_r  : float value with the infinitesimal dr approximation inside the integral.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the tau parameter in the shifted-geometric function.
# Returns:
# ---No return value. The probability distribution function values are stored in the F_vals array.
@boundscheck(False)
@wraparound(False)
def F_shifted_geometric_eval( np.ndarray[float,ndim=1,mode="c"] F_vals, \
                              int N_points, float delta_r, float f, float tau ):
   cdef:
      int i,j
      double S,arg,fact,Ei_f,num1,num2 #,gamma
   Ei_r        = np.zeros((N_points,),dtype=np.float32)
   Ei_f        = 0.
   for i in range(0,N_points):
         num1     = (float(i) + 0.5) * delta_r
         arg      = f / (1.0 + (tau*num1))
         Ei_r[ i ]= f_sp.Ei( arg )
   Ei_f        = f_sp.Ei( f )
   for i in range(0,N_points):
      num1     = (float( i ) + 0.5) * delta_r
      arg      = f / (1.0 + (tau*num1))
      fact     = f * exp(-f)
      num1     = (1. + (tau*num1)) * exp( arg - f ) - 1.0 + ( fact * ( Ei_f - Ei_r[ i ] ) )
      num2     = (1. + ( tau )) * exp( f * ( (1.0 / (1.0 + tau)) - 1.0 ) ) - 1.0 + ( fact * ( Ei_f - Ei_r[ N_points-1 ] ) )
      F_vals[ i ] = num1 / num2
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to apply the Gibbs Sampling Method to draw a histogram of the samples from the
# discrete binary version of the polylogarithmic exponential distribution.
# Parameters:
# ---n_samples: integer array where the N_SAMP random samples are to be stored.
#               Each sample is an integer from 0 to N, indicating the number of active
#               binary neurons in the population.
# ---NITE     : integer value with the number of simulations per sample (for burn-in period).
# ---N        : integer value with the number of neurons in the population.
# ---N_SAMP   : integer value with the number of random samples to store in n_samples.
# ---M_TERMS  : integer value with the number of terms to consider in the alternating series
#               for the case of m > 1. This value should be in [3,4,5,...,N].
# ---F        : float value with the sparsity inducing parameter.
# ---m        : integer value indicating the order of the polylogarithmic function.
# Returns:
# ---No return value. The histogram is stored in the n_samples array.
@boundscheck(False)
@wraparound(False)
def GibbsSampling_polylogarithmic_hist( np.ndarray[int,ndim=1,mode="c"] n_samples, \
                                        int NITE, int N, int N_SAMP, int M_TERMS, float F, int m ):
   cdef:
      int r,j,n
   cdef int[:] X_SAMPLES = np.zeros((N_SAMP*N,),dtype=np.int32)
   printf("Sampling from %d-dimensional binary distribution\n",N)
   # Parallel simulation of each sample
   for r in prange(0,N_SAMP,nogil=True,schedule='static',num_threads=4):
      # Serial ordered computation
      c_polylogarithmic_samp( r, NITE, &X_SAMPLES[0], N,M_TERMS, F,m )
   # Compute number of active neurons per sample and store
   for r in range(0,N_SAMP):
      n       = 0
      for j in range(0,N):
         n    = n + X_SAMPLES[ r*N + j ]
      n_samples[ r ] = n
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to apply the Gibbs Sampling Method to draw samples from the discrete binary
# version of the polylogarithmic exponential distribution.
# Parameters:
# ---X_SAMPLES: integer array where the N_SAMP random samples are to be stored.
#               Each sample is a binary vector of size N (stored consecutively).
# ---NITE     : integer value with the number of simulations per sample (for burn-in period).
# ---N        : integer value with the number of neurons in the population.
# ---N_SAMP   : integer value with the number of random samples to store in n_samples.
# ---M_TERMS  : integer value with the number of terms to consider in the alternating series
#               for the case of m > 1. This value should be in [3,4,5,...,N].
# ---F        : float value with the sparsity inducing parameter.
# ---m        : integer value indicating the order of the polylogarithmic function.
# Returns:
# ---No return value. The random samples are stored in the X_SAMPLES array.
@boundscheck(False)
@wraparound(False)
def GibbsSampling_polylogarithmic( np.ndarray[int,ndim=1,mode="c"] X_SAMPLES, \
                                   int NITE, int N, int N_SAMP, int M_TERMS, float F, int m ):
   cdef:
      int r,j,n
   printf("Sampling from %d-dimensional binary distribution\n",N)
   # Parallel simulation of each sample
   for r in prange(0,N_SAMP,nogil=True,schedule='static',num_threads=4):
      # Serial ordered computation
      c_polylogarithmic_samp( r, NITE, &X_SAMPLES[0], N,M_TERMS, F,m )
   # Compute number of active neurons per sample and store
   for r in range(0,N_SAMP):
      n       = 0
      for j in range(0,N):
         n    = n + X_SAMPLES[ r*N + j ]
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to apply the Gibbs Sampling Method to draw a histogram of the samples from the
# discrete binary version of the shifted-geometric exponential distribution.
# Parameters:
# ---n_samples: integer array where the N_SAMP random samples are to be stored.
#               Each sample is an integer from 0 to N, indicating the number of active
#               binary neurons in the population.
# ---NITE     : integer value with the number of simulations per sample (for burn-in period).
# ---N        : integer value with the number of neurons in the population.
# ---N_SAMP   : integer value with the number of random samples to store in r_samples.
# ---F        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric exponential tau parameter.
# Returns:
# ---No return value. The random samples are stored in the n_samples array.
@boundscheck(False)
@wraparound(False)
def GibbsSampling_shifted_geometric_hist( np.ndarray[int,ndim=1,mode="c"] n_samples, \
                                          int NITE, int N, int N_SAMP, float F, float tau ):
   cdef:
      int r,j,n
   cdef int[:] X_SAMPLES = np.zeros((N_SAMP*N,),dtype=np.int32)
   printf("Sampling from %d-dimensional binary distribution\n",N)
   # Parallel simulation of each sample
   for r in prange(0,N_SAMP,nogil=True,schedule='static',num_threads=4):
      # Serial ordered computation
      c_shifted_geometric_samp( r, NITE, &X_SAMPLES[0], N, F, tau )
   # Compute number of active neurons per sample and store
   for r in range(0,N_SAMP):
      n       = 0
      for j in range(0,N):
         n    = n + X_SAMPLES[ r*N + j ]
      n_samples[ r ] = n
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to apply the Gibbs Sampling Method to draw samples from the discrete binary
# version of the shifted-geometric exponential distribution.
# Parameters:
# ---X_SAMPLES: integer array where the N_SAMP random samples are to be stored.
#               Each sample is a binary vector of size N (stored consecutively).
# ---NITE     : integer value with the number of simulations per sample (for burn-in period).
# ---N        : integer value with the number of neurons in the population.
# ---N_SAMP   : integer value with the number of random samples to store in r_samples.
# ---F        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric exponential tau parameter.
# Returns:
# ---No return value. The random samples are stored in the n_samples array.
@boundscheck(False)
@wraparound(False)
def GibbsSampling_shifted_geometric( np.ndarray[int,ndim=1,mode="c"] X_SAMPLES, \
                                     int NITE, int N, int N_SAMP, float F, float tau ):
   cdef:
      int r,j,n
   printf("Sampling from %d-dimensional binary distribution\n",N)
   # Parallel simulation of each sample
   for r in prange(0,N_SAMP,nogil=True,schedule='static',num_threads=4):
      # Serial ordered computation
      c_shifted_geometric_samp( r, NITE, &X_SAMPLES[0], N, F, tau )
   # Compute number of active neurons per sample and store
   for r in range(0,N_SAMP):
      n       = 0
      for j in range(0,N):
         n    = n + X_SAMPLES[ r*N + j ]
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to apply the Inverse Transform Method to draw samples from continuous
# probability distributions (polylogarithmic exponential and shifted-geometric exponential).
# Parameters:
# ---r_samples: float array where the N_SAMP random samples are to be stored.
# ---F_vals   : float array where each of the N_points values of the discretized probability
#               distribution function are stored.
# ---N_SAMP   : integer value with the number of random samples to store in r_samples.
# ---N_points : integer value with the number of discrete points in F_vals.
# ---delta_r  : float value with the infinitesimal dr approximation inside the integral.
# Returns:
# ---No return value. The random samples are stored in the r_samples array.
@boundscheck(False)
@wraparound(False)
def InverseTransform( np.ndarray[float,ndim=1,mode="c"] r_samples,\
                      np.ndarray[float,ndim=1,mode="c"] F_vals,\
                      int N_SAMP, int N_points, float delta_r ):
   cdef:
      int i,ind
      float r,r_p,U
   printf("Sampling...\n")
   for i in range(0,N_SAMP+1):
      r      = 0.
      r_p    = 0.
      U      = float(rand())/ float(RAND_MAX)
      ind    = int( r / delta_r )
      while( r < 1.0 and F_vals[ ind ] < U ):
         r_p = r
         r   = r + delta_r
         ind = int( r / delta_r )
         if ind >= N_points :
            ind = N_points - 1
            break
      if r < 1.0 :
         r_samples[i] = r_p
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Declaration of all c code functions called from within this Cython file.
# The header function declaration file must be located in the extern from
# "path_to_c_header_file.h" defined below.
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
cdef extern from "c/c_functions_sampling.h" nogil:
   
   void c_initRandom()
   void c_polylogarithmic_samp( int r, int NITE, int *X,\
                                int N, int M_TERMS, float F, int m )
   void c_shifted_geometric_samp( int r, int NITE, int *X,\
                                  int N, float F, float tau )
