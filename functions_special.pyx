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
   ei        = 0.5772156649 + log( x )
   #for k in range(1,50):
   for k in range(1,150):
      term   = x / np.double(k*k)
      for i in range(1,k):
         term= term * ( x / np.double(i))
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
   x_2       = x * x
   erfi_v    = x
   for k in range(1,150):
      term   = x / np.double( 2*k + 1 )
      for i in range(1,k+1):
         term= term * ( x_2 / np.double(i) )
      erfi_v = erfi_v + term
   return (2.0 * erfi_v / sqrt(M_PI))