import numpy as np
import functions_special as f_sp
from scipy import special as sp
from libc.stdio cimport printf
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs,M_PI

cimport numpy as np
cimport cython


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the probability density function (PDF) for the continuous population rate
# considering the polylogarithmic exponential PDF.
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---m        : integer value with the positive integer order of the polylogarithmic function.
# ---M        : integer value with the number of terms at which to truncate the infinite series
#               for evaluation of the polylogarithmic case m>1.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def polylogarithmic_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                         np.ndarray[float,ndim=1,mode="c"] r,\
                         int npoints, float f, int m, int M ):
   cdef:
      int pi,j
      double Z,Lim
   if m == 1 :
      for pi in range(0,npoints):
         p[ pi ] = exp( -f * log( 1. + r[ pi ] ) )
      if f != 1. :
         Z       = ( pow(2.,1.-f) - pow(1.,1.-f) ) / ( 1. - f )
      else :
         Z       = log( 2.0 )
   else :
      Z          = 0.
      for pi in range(0,npoints):
         Lim     = 0
         for j in range(1,M+1):
            Lim  = Lim + ( pow( -r[ pi], np.double(j) ) / pow( np.double(j), np.double(m) ) )
         p[ pi ] = exp( f * Lim )
         Z       = Z + p[ pi ]
      Z          = Z * ( 1. / np.double(npoints) ) # multiply by limiting dr for integration
   for pi in range(0,npoints):
      p[ pi ] = p[ pi ] / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the probability density function (PDF) for the continuous population rate
# considering the shifted-geometric exponential PDF.
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the tau parameter in the shifted-geometric function.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def shifted_geometric_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                           np.ndarray[float,ndim=1,mode="c"] r,\
                           int npoints, float f, float tau ):
   cdef:
      int pi
      double Z,Ei_1,Ei_0,x_1
   for pi in range(0,npoints):
      p[ pi ] = exp( f*( (1. / (1. + (tau * r[ pi ]) )) - 1. ) )
   
   x_1        = f / (1. + (tau ))
   #Ei_1       = f_sp.Ei( x_1 )
   #Ei_0       = f_sp.Ei( f )
   Ei_1       = sp.expi( x_1 )
   Ei_0       = sp.expi( f )
   
   Z          = ((1. + tau) / tau)*exp( x_1 - f ) - ((f*exp(-f) / tau) * Ei_1)
   Z          = Z - ( (1. / tau) - ((f*exp(-f) / tau) * Ei_0) )
   #printf("Z =%f\n",Z)
   for pi in range(0,npoints):
      p[ pi ] = p[ pi ] / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the probability density function (PDF) for the continuous population rate
# considering the first-order interactions exponential PDF.
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def first_ord_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                   np.ndarray[float,ndim=1,mode="c"] r, int npoints, float f ):
   cdef:
      int pi
      double Z
   for pi in range(0,npoints):
      p[ pi ] = exp( -f* r[ pi ] )
   
   Z          = (1.0 - exp(-f)) / f
   for pi in range(0,npoints):
      p[ pi ] = p[ pi ] / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the probability density function (PDF) for the continuous population rate
# considering the second-order interactions exponential PDF.
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f1       : float value with the sparsity inducing parameter.
# ---f2       : float value with the cuadratic coefficient parameter.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def second_ord_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                    np.ndarray[float,ndim=1,mode="c"] r, int npoints, float f1, float f2 ):
   cdef:
      int pi,N
      double Z,er_0,er_1,K,s,dr,eps
   for pi in range(0,npoints):
      p[ pi ] = exp( (f1 * r[ pi ]) + (f2 * r[pi] * r[pi] )  )
   
   s          = f1 / ( 2.0 * sqrt( fabs(f2) ) )
   if fabs( s ) <= 5.0 :
      # Compute analytic normalization constant
      # only when f1 is not too far away from f2
      # (otherwise the difference between erfi or erf
      #  functions becomes numerically unstable)
      if f2 > 0. :
         K    = ( sqrt( M_PI ) / (2.0 * sqrt(f2)) ) * exp( -(s*s) )
         er_1 = sp.erfi( s + sqrt(f2) )
         er_0 = sp.erfi( s )
         Z    = K * ( er_1 - er_0 )
      else :
         K    = ( sqrt( M_PI ) / (2.0 * sqrt( fabs(f2) )) ) * exp( (s*s) )
         er_1 = sp.erf( -s + sqrt( fabs(f2) ) )
         er_0 = sp.erf( -s )
         Z    = K * ( er_1 - er_0 )
   else :
      # Compute discretized version of normalization
      # constant
      eps     = 1e-15
      N       = 10000
      r_do    = np.asarray( np.arange(0,1.0 + float(1. / float(N)), float(1. / float(N)) ) , dtype=np.float32)
      r_do[0] = eps
      r_do[N] = 1.0 - eps
      r_do    = r_do.copy(order='C')
      dr      = 1. / float(N)
      Z       = 0.
      for pi in range(0,N+1):
         Z    = Z + ( exp(  ( (f1 * r_do[ pi ]) + (f2 * r_do[ pi ] * r_do[ pi ] )  )  ) * dr )
   for pi in range(0,npoints):
      p[ pi ] = p[ pi ] / Z
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the first-order approximation of the polylogarithmic function to
# approximate the polylogarithmic exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def first_order_poly_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                          np.ndarray[float,ndim=1,mode="c"] r,
                          int npoints, float f ):
   cdef:
      int pi
   for pi in range(0,npoints):
      p[ pi ] = f * exp( -f * r[pi] ) / ( 1. - exp(-f) )
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the second-order approximation of the polylogarithmic function to
# approximate the polylogarithmic exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---m        : integer value with the positive integer order of the polylogarithmic function.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def second_order_poly_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                          np.ndarray[float,ndim=1,mode="c"] r,
                          int npoints, float f, int m ):
   cdef:
      int pi
      float delta_r,Z,r_i
   delta_r = 1. / float(npoints)
   Z       = 0.
   for pi in range(0,npoints-1):
      r_i  = (r[pi] + r[pi+1]) / 2.
      Z    = Z + exp( -(f*r_i) + (f*pow(r_i,2) / pow(2.,float(m))) )
   Z       = Z * delta_r
   for pi in range(0,npoints):
      p[ pi ] = exp( -(f*r[pi]) + (f*pow(r[pi],2) / pow(2.,float(m))) ) / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the third-order approximation of the polylogarithmic function to
# approximate the polylogarithmic exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---m        : integer value with the positive integer order of the polylogarithmic function.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def third_order_poly_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                          np.ndarray[float,ndim=1,mode="c"] r,
                          int npoints, float f, int m ):
   cdef:
      int pi
      float delta_r,Z,r_i
   delta_r = 1. / float(npoints)
   Z       = 0.
   for pi in range(0,npoints-1):
      r_i  = (r[pi] + r[pi+1]) / 2.
      Z    = Z + exp( -(f*r_i) + (f*pow(r_i,2) / pow(2.,float(m))) -(f*pow(r_i,3) / pow(3.,float(m))) )
   Z       = Z * delta_r
   for pi in range(0,npoints):
      p[ pi ] = exp( -(f*r[pi]) + (f*pow(r[pi],2) / pow(2.,float(m))) -(f*pow(r[pi],3) / pow(3.,float(m))) ) / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the fourth-order approximation of the polylogarithmic function to
# approximate the polylogarithmic exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---m        : integer value with the positive integer order of the polylogarithmic function.
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def fourth_order_poly_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                           np.ndarray[float,ndim=1,mode="c"] r,
                           int npoints, float f, int m ):
   cdef:
      int pi
      float delta_r,Z,r_i
   delta_r = 1. / float(npoints)
   Z       = 0.
   for pi in range(0,npoints-1):
      r_i  = (r[pi] + r[pi+1]) / 2.
      Z    = Z + exp( -(f*r_i) + (f*pow(r_i,2) / pow(2.,float(m))) -(f*pow(r_i,3) / pow(3.,float(m))) + (f*pow(r_i,4) / pow(4.,float(m))) )
   Z       = Z * delta_r
   for pi in range(0,npoints):
      p[ pi ] = exp( -(f*r[pi]) + (f*pow(r[pi],2) / pow(2.,float(m))) -(f*pow(r[pi],3) / pow(3.,float(m))) + (f*pow(r[pi],4) / pow(4.,float(m))) ) / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the first-order approximation of the shifted-geometric function to
# approximate the shifted-geometric exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric function parameter tau in (0,1).
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def first_order_s_geom_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                            np.ndarray[float,ndim=1,mode="c"] r,
                            int npoints, float f, float tau ):
   cdef:
      int pi
   for pi in range(0,npoints):
      p[ pi ] = f * tau * exp( -f*tau * r[pi] ) / ( 1. - exp(-f*tau) )
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the second-order approximation of the shifted-geometric function to
# approximate the shifted-geometric exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric function parameter tau in (0,1).
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def second_order_s_geom_pdf( np.ndarray[float,ndim=1,mode="c"] p,\
                             np.ndarray[float,ndim=1,mode="c"] r,
                             int npoints, float f, float tau ):
   cdef:
      int pi
      float delta_r,Z,r_i
   delta_r = 1. / float(npoints)
   Z       = 0.
   for pi in range(0,npoints-1):
      r_i  = (r[pi] + r[pi+1]) / 2.
      Z    = Z + exp( -(f*tau*r_i) + (f*tau*tau*r_i*r_i) )
   Z       = Z * delta_r
   for pi in range(0,npoints):
      p[ pi ] = exp( -(f*tau*r[pi]) + (f*tau*tau*pow(r[pi],2)) ) / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the third-order approximation of the shifted-geometric function to
# approximate the shifted-geometric exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric function parameter tau in (0,1).
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def third_order_s_geom_pdf(  np.ndarray[float,ndim=1,mode="c"] p,\
                             np.ndarray[float,ndim=1,mode="c"] r,
                             int npoints, float f, float tau ):
   cdef:
      int pi
      float delta_r,Z,r_i
   delta_r = 1. / float(npoints)
   Z       = 0.
   for pi in range(0,npoints-1):
      r_i  = (r[pi] + r[pi+1]) / 2.
      Z    = Z + exp( -(f*tau*r_i) + (f*tau*tau*r_i*r_i) -(f*pow(tau*r_i,3)) )
   Z       = Z * delta_r
   for pi in range(0,npoints):
      p[ pi ] = exp( -(f*tau*r[pi]) + (f*tau*tau*pow(r[pi],2)) -(f*pow(tau*r[pi],3)) ) / Z
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the third-order approximation of the shifted-geometric function to
# approximate the shifted-geometric exponential probability density function (PDF).
# Parameters:
# ---p        : float array where each of the npoints values of the discretized probability
#               density function will be stored.
# ---r        : float array with the population rate values in [0,1] to evaluate.
# ---npoints  : integer value with the number of discrete points to evaluate.
# ---f        : float value with the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric function parameter tau in (0,1).
# Returns:
# ---No return value. The probability density function values are stored in the p array.
@boundscheck(False)
@wraparound(False)
def fourth_order_s_geom_pdf(  np.ndarray[float,ndim=1,mode="c"] p,\
                             np.ndarray[float,ndim=1,mode="c"] r,
                             int npoints, float f, float tau ):
   cdef:
      int pi
      float delta_r,Z,r_i,tau_ri
   delta_r   = 1. / float(npoints)
   Z         = 0.
   for pi in range(0,npoints-1):
      r_i    = (r[pi] + r[pi+1]) / 2.
      tau_ri = tau*r_i
      Z      = Z + exp( -(f*tau_ri) + (f*pow(tau_ri,2)) -(f*pow(tau_ri,3)) + (f*pow(tau_ri,4)) )
   Z         = Z * delta_r
   for pi in range(0,npoints):
      tau_ri = tau*r[pi]
      p[ pi ]= exp( -(f*tau_ri) + (f*pow(tau_ri,2)) -(f*pow(tau_ri,3)) + (f*pow(tau_ri,4)) ) / Z
   return None
