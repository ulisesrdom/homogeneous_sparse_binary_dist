import numpy as np
from libc.stdio cimport printf
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs

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
      double ei,fact
   ei        = 0.5772156649 + log( x )
   for k in range(1,50):#45):
      # fact[k] = factorial( k )
      fact   = np.double(k)
      for i in range(2,k):
         fact = fact * (np.double(i))
      #printf("factorial at k=%i : %f\n",k,fact)
      ei     = ei + ( pow( x, np.double(k) ) / ( (np.double(k)) * fact ) )
   return ei