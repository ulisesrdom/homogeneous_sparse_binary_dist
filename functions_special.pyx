import numpy as np
from libc.stdio cimport printf
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs,M_PI

cimport numpy as np
cimport cython

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the exponential integral special function for a real scalar argument.
# Parameters:
# ---x        : float value with the real value to evaluate.
# Returns:
# ---float scalar with the function evaluation.
@boundscheck(False)
@wraparound(False)
def Ei( double x ):
   cdef:
      int k,i
      double ei,term
   cdef double[:] ser_k = np.zeros((400,),dtype=np.float64)
   ei        = 0.5772156649 + log( x )
   
   ser_k[0]  = x
   for i in range(1,400):
    ser_k[i] = ( x / np.double(i+1) ) * ser_k[ i-1 ]
   
   for k in range(1,400):
      term   = (ser_k[ k-1] / np.double(k))
      #printf("term at k=%i : %f\n",k,term)
      ei     = ei + term
   return ei

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the imaginary error function as defined by its Maclaurin series
# for a real scalar argument.
# Parameters:
# ---x        : float value with the real value to evaluate.
# Returns:
# ---float scalar with the function evaluation.
@boundscheck(False)
@wraparound(False)
def erfi( double x ):
   cdef:
      int k,i
      double erfi_v,term,x_2
   cdef double[:] ser_k = np.zeros((300,),dtype=np.float64)
   x_2       = x * x
   ser_k[0]  = x_2
   for i in range(1,300):
    ser_k[i] = ( x_2 / np.double(i+1) ) * ser_k[ i-1 ]
   erfi_v    = x
   for k in range(1,300):
      term   = ( x / np.double( 2*k + 1 ) ) * ser_k[ k-1 ]
      erfi_v = erfi_v + term
   return (2.0 * erfi_v / sqrt(M_PI))