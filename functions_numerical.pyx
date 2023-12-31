import numpy as np
import functions_sampling as f_samp
from libc.stdio cimport printf
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs

cimport numpy as np
cimport cython

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute an array with ordered values of the factorial of positive integer numbers.
# Parameters:
# ---fact     : double array where the ordered factorial numbers are to be stored in
#               ascending order. The factorials to evaluate are from 1 to N.
# ---N        : integer value with the maximum integer to evaluate in the array of factorials.
# Returns:
# ---No return value. The factorial values are stored in the fact array.
@boundscheck(False)
@wraparound(False)
def factorial_array( np.ndarray[double,ndim=1,mode="c"] fact, int N ):
   cdef:
      int k
   # Compute factorials
   fact[ 0 ]      = 1.0
   for k in range(2,N+1):
      fact[ k-1 ] = np.double(k) * fact[ k-2 ]
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute a single multinomial coefficient.
# Parameters:
# ---k        : integer m-dimensional array storing the integer indices in the 
#               denominator of the multinomial coefficient.
# ---m        : integer with the number of integer indices for the multinomial
#               coefficient denominator.
# ---fact     : double array storing pre-calculated factorial numbers with
#               ascending order. The dimension of this array must be at
#               least as big as the summation over the elements of k.
# Returns:
# ---double M_C with the multinomial coefficient value.
@boundscheck(False)
@wraparound(False)
def multinomial( np.ndarray[int,ndim=1,mode="c"] k, int m,\
                 np.ndarray[double,ndim=1,mode="c"] fact ):
   cdef:
      int i,ti,ind1
      double temp1,temp2,M_C
   M_C       = 1.0
   for ti in range(1,m):
      ind1   = 0
      for i in range(0,ti+1):
         ind1 = ind1 + k[ i ]
      temp1  = fact[ ind1-1 ]
      ind1   = 0
      for i in range(0,ti):
         ind1 = ind1 + k[ i ]
      temp2  = fact[ ind1-1 ]
      M_C    = M_C  *  ( temp1 / ( np.double(k[ti]) * temp2 ) )
   return M_C

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the summation over multinomial coefficients with strictly
# positive integer indices in the denominator of each single multinomial coefficient.
# Parameters:
# ---m        : integer with the number of integer indices for the multinomial
#               coefficient denominator.
# ---j        : value of the summation over the integer indices in each
#               multinomial coefficient denominator that must hold for
#               each coefficient.
# ---fact     : double array storing pre-calculated factorial numbers with
#               ascending order. The dimension of this array must be at
#               least as big as j.
# Returns:
# ---double MSUM with the summation over the multinomial coefficients.
@boundscheck(False)
@wraparound(False)
def M_coeff_sum( int m, int j, \
                 np.ndarray[double,ndim=1,mode="c"] fact ):
   cdef:
      int ind,i,S
      double MSUM
   k          = np.ones((m,),dtype=np.int32)
   k          = k.copy(order='C')
   if j == m :
      MSUM    = fact[ j-1 ]
   else:
      ind     = 0
      MSUM    = 0.
      while ind < m :
         # Compute current sum and test if it equals j to add coefficient
         S    = 0
         for i in range(0,m):
            S = S + k[ i ]
         if S == j :
            MSUM = MSUM + multinomial( k, m, fact )
         # Advance current coordinate value
         k[ind]  = k[ind] + 1
         # Advance coordinate index
         if S > j or k[ind] >= j :
            k[ind] = 1
            ind    = ind + 1
   return MSUM

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the canonical theta coordinates for the homogeneous model.
# Parameters:
# ---THETA    : float N-dimensional array where the canonical theta coordinates
#               are to be stored.
# ---fact     : double array storing pre-calculated factorial numbers with
#               ascending order. The dimension of this array must be at
#               least as big as N.
# ---N        : integer with the number of theta coordinates (which is
#               also the number of neurons in the population).
# ---DISTR_TYPE: integer to select the distribution type to consider. The options are:
#               polylogarithmic exponential (1); shifted-geometric exponential (2).
# ---f        : float value with the sparsity inducing parameter for the distributions.
# ---m        : integer value with the positive integer order of the polylogarithmic function
#               for DISTR_TYPE=1.
# ---tau      : float value with the tau parameter in the shifted-geometric function
#               for DISTR_TYPE=2.
# Returns:
# ---No return value. The theta coordinates are stored in the THETA array.
@boundscheck(False)
@wraparound(False)
def obtain_theta( np.ndarray[float,ndim=1,mode="c"] THETA, \
                  np.ndarray[double,ndim=1,mode="c"] fact, \
                  int N, int DISTR_TYPE, float f, int m, float tau ):
   cdef:
      int j,l
      float S, MC
   cdef float[:] C = np.zeros((N,),dtype=np.float32)
   
   # Set coefficients depending on the distribution type
   if DISTR_TYPE == 1 :
      # Set coefficients for polylogarithmic function
      for l in range(1,N+1):
         C[l-1] = 1.0 /  pow( float(l) , float(m) )
   else:
      # Set coefficients for shifted-geometric function
      for l in range(1,N+1):
         C[l-1] = pow( tau , float(l) )
   
   # Obtain theta_1
   S           = 0.0
   for l in range(1,N+1):
      S        = S + (  pow( -1., float(l) ) * f * C[l-1] / pow( float(N), float(l) )  )
   THETA[0]    = S
   # Obtain theta_2, theta_3, ..., theta_N
   for j in range(1,N):
      S        = 0.0
      for l in range(j+1,N+1):
         MC    = M_coeff_sum( j+1, l, fact )
         S     = S + ( pow( -1., float(l) ) * f * C[l-1] * MC / pow( float(N), float(l) ) )
      THETA[j] = S
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute the polynomial Q_N(r_N ; theta_N) using the Stirling numbers of
# the first kind representation.
# Parameters:
# ---Q        : float N-dimensional array storing the polynomial values evaluated at
#               N r_N points. Each point to evaluate multiplied by N yields the
#               integer N * r_N = n.
# ---FACT_k   : double (N-1)-dimensional array storing the factorials from 1 to N-1
#               in ascending order.
# ---THETA    : float (N-1)-dimensional array storing the canonical homogeneous theta
#               coordinates that determine the higher order interactions.
# ---N        : integer value with the number of points to evaluate, equivalent to the
#               number of neurons in the population plus one.
# Returns:
# ---No return value. The polynomial values are stored in the Q array.
@boundscheck(False)
@wraparound(False)
def Q_polynomial( np.ndarray[float,ndim=1,mode="c"] Q,\
                  np.ndarray[float,ndim=1,mode="c"] THETA, \
                  np.ndarray[double,ndim=1,mode="c"] FACT_k, int N ):
   cdef:
      int i,k,m,ind,ind_aux,ni
      double S
   if N > 21 :
      N = 21
      printf("Q_polynomial:: N is limited to 21.\n")
   cdef double[:] POW_n_k = np.zeros((N*N,),dtype=np.float64)
   cdef double[:] SS1_k_m = np.zeros((N*N,),dtype=np.float64)
   
   # Pre-compute powers for faster computation
   for m in range(0,N):
      for k in range(0,N):
         ind            = m*N + k
         POW_n_k[ ind ] = pow( np.double( m ), np.double( k ) )
   # Pre-compute Stirling numbers of the 1st kind for faster computation
   for k in range(0,N):
      ind              = k * N  +  k
      SS1_k_m[ ind ]   = 1.0
   for k in range(1,N):
      for m in range(1,N):
         if m != k :
            ind           = k * N  +  m
            ind_aux       = (k-1) * N  +  m
            SS1_k_m[ ind ]= -np.double(k-1) * SS1_k_m[ ind_aux ]
            ind_aux       = (k-1) * N  +  m-1
            SS1_k_m[ ind ]= SS1_k_m[ ind ] + SS1_k_m[ ind_aux ]
   # Finish computation of Q_N( r_N ; theta_N ) polynomial
   Q[0]      = 0.0
   for ni in range(1,N):
      Q[ni]  = 0.0
      for k in range(1,ni+1):
         S   = 0.0
         for m in range(0,k+1):
            ind  = k * N  +  m
            S    = S + (SS1_k_m[ ind ] * POW_n_k[ ni * N + m ] / FACT_k[ k-1 ] )
         Q[ni]   = Q[ni] + ( THETA[ k-1 ] * S )
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the polylogarithmic exponential distr. model to a given dataset of binary
# spikes (assummed to come from spontaneous neural activity).
# Parameters:
# ---X        : float flattened array with Ns binary samples from real spiking data, each of
#               size N.
# ---N        : integer value with the size of the neural population in each sample.
# ---Ns       : integer value with the number of samples in the binary data.
# ---M_TERMS  : integer value with the number of terms to consider in the alternating series
#               for the case of m > 1. This value should be in [3,4,5,...,N].
# ---M        : integer value with the number of Gibbs samples to produce for the derivative
#               of the normalization function with respect to the f parameter.
# ---eta      : float value with the learning rate for locally optimizing over the f parameter.
# ---f_init   : float value with the initial value for the sparsity inducing parameter.
# ---m_min    : integer value with the minimum value for the grid-search range of the integer
#               order of the polylogarithmic function.
# ---m_max    : integer value with the maximum value for the grid-search range of the integer
#               order of the polylogarithmic function.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Pair of fitted parameters: f and m, where f is the sparsity inducing parameter and
#    m is the integer order parameter of the polylogarithmic function.
@boundscheck(False)
@wraparound(False)
def model_fit_polylogarithmic(  np.ndarray[int,ndim=1,mode="c"] X, int N, int Ns, int M, int M_TERMS,
                                float eta, float f_init, int m_min, int m_max, int MAX_ITE, int T ):
   cdef:
      int ite,t,m,m_p,i
      float f,s1,s2
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_wrt_f     = np.zeros((M,),dtype=np.float32)
   cdef np.ndarray[int, ndim=1, mode='c'] X_SAMPLES
   X_SAMPLES                      = np.zeros((M*N,),dtype=np.int32, order='C')
   # Initialize parameters
   f         = f_init
   m         = m_min
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      # Locally optimize for the sparsity inducing parameter f given m
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,M):
            lZ_der_wrt_f[i] = 0.0
         # Parallel computation of first part of log-likelihood derivative
         for i in prange(0,Ns,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_poly_wrt_f_part( i, &ll_batch[0], m, &X[0], N )
         # Collapse parallel batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         # Obtain M Gibbs samples for the second part of the derivative
         f_samp.GibbsSampling_polylogarithmic( X_SAMPLES, 11, N, M, M_TERMS, f, m )
         
         # Parallel computation of second part of log-likelihood derivative
         for i in prange(0,M,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_poly_wrt_f_part( i, &lZ_der_wrt_f[0], m, &X_SAMPLES[0], N )
         # Collapse parallel batch results
         s2  = 0.0
         for i in range(0,M):
           s2= s2 + lZ_der_wrt_f[ i ]
         s2  = s2 / float( M )
         f   = f + (eta*(-s1+s2))
      
      # Locally optimize (grid search) for the m polylogarithmic parameter given f
      # ---------------------------------------------------------------------------
      s2     = float('-inf')
      for m_p in range(m_min,m_max+1):
         for i in range(0,Ns):
            ll_batch[i] = 0.0
         # Parallel computation of the partial log-likelihood for each sample
         for i in prange(0,Ns,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_log_likelihood_poly_part_i( i, &X[0], &ll_batch[0], f,m_p,  N )
         # Collapse parallel batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         if s1 > s2 :
           s2= s1
           m = m_p
      print("model_fit_polylogarithmic:: iteration {}, f={}, m={}".format(ite,f,m))
   return f,m

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the shifted-geometric exponential distr. model to a given dataset of binary
# spikes (assummed to come from spontaneous neural activity).
# Parameters:
# ---X        : float flattened array with Ns binary samples from real spiking data, each of
#               size N.
# ---N        : integer value with the size of the neural population in each sample.
# ---Ns       : integer value with the number of samples in the binary data.
# ---M        : integer value with the number of Gibbs samples to produce for the derivative
#               of the normalization function with respect to the f parameter.
# ---eta      : float value with the learning rate for locally optimizing the f-parameter.
# ---eta_tau  : float value with the learning rate for locally optimizing the tau-parameter.
# ---f_init   : float value with the initial value for the sparsity inducing parameter.
# ---tau_init : float value with the initial value for the shifted-geometric tau parameter.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Pair of fitted parameters: f and tau, where f is the sparsity inducing parameter and
#    tau is the shifted-geometric function parameter.
@boundscheck(False)
@wraparound(False)
def model_fit_shifted_geom( np.ndarray[int,ndim=1,mode="c"] X, int N, int Ns, int M, \
                            float eta, float eta_tau, float f_init, float tau_init, int MAX_ITE, int T ):
   cdef:
      int ite,t,i
      float f,s1,s2,tau
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_batch     = np.zeros((M,),dtype=np.float32)
   cdef np.ndarray[int, ndim=1, mode='c'] X_SAMPLES
   X_SAMPLES                      = np.zeros((M*N,),dtype=np.int32, order='C')
   # Initialize parameters
   f         = f_init
   tau       = tau_init
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      # Locally optimize for the sparsity inducing parameter f given tau
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,M):
            lZ_der_batch[i] = 0.0
         # Parallel computation of first part of log-likelihood derivative
         for i in prange(0,Ns,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_sg_wrt_f_part( i, &ll_batch[0], tau, &X[0], N )
         # Collapse parallel batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         # Obtain M Gibbs samples for the second part of the derivative
         f_samp.GibbsSampling_shifted_geometric( X_SAMPLES, 11, N, M, f, tau )
         
         # Parallel computation of second part of log-likelihood derivative
         for i in prange(0,M,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_sg_wrt_f_part( i, &lZ_der_batch[0], tau, &X_SAMPLES[0], N )
         # Collapse parallel batch results
         s2  = 0.0
         for i in range(0,M):
           s2= s2 + lZ_der_batch[ i ]
         s2  = s2 / float( M )
         f   = f + (eta*(-s1+s2))
         if f < 0.0 :
            f = 0.0
      
      # Locally optimize for the tau shifted-geometric parameter given f
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,M):
            lZ_der_batch[i] = 0.0
         # Parallel computation of first part of log-likelihood derivative
         for i in prange(0,Ns,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_sg_wrt_tau_part( i, &ll_batch[0], f,tau, &X[0], N )
         # Collapse parallel batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         # Obtain M Gibbs samples for the second part of the derivative
         f_samp.GibbsSampling_shifted_geometric( X_SAMPLES, 11, N, M, f, tau )
         
         # Parallel computation of second part of log-likelihood derivative
         for i in prange(0,M,nogil=True,schedule='static',num_threads=4):
            # Serial ordered computation
            c_der_ll_sg_wrt_tau_part( i, &lZ_der_batch[0], f,tau, &X_SAMPLES[0], N )
         # Collapse parallel batch results
         s2  = 0.0
         for i in range(0,M):
           s2= s2 + lZ_der_batch[ i ]
         s2  = s2 / float( M )
         tau = tau + (eta_tau*(-s1+s2))
         if tau < 0 :
            tau = 0.
         if tau > 1.0 :
            tau = 1.0
      
      print("model_fit_shifted_geom:: iteration {}, f={}, tau={}".format(ite,f,tau))
   return f,tau

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Declaration of all c code functions called from within this Cython file.
# The header function declaration file must be located in the extern from
# "path_to_c_header_file.h" defined below.
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
cdef extern from "c/c_functions_numerical.h" nogil:
   
   void c_der_ll_poly_wrt_f_part( int i, float *ll_batch, int m, int *X, int N )
   void c_log_likelihood_poly_part_i( int i, int *X, float *ll_batch, float f, float m,  int N )
   void c_der_ll_sg_wrt_f_part( int i, float *ll_batch, float tau, int *X, int N )
   void c_der_ll_sg_wrt_tau_part( int i, float *ll_batch, float f, float tau, int *X, int N )
