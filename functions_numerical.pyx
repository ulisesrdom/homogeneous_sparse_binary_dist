import numpy as np
import functions_sampling as f_samp
import functions_generic as f_gen
import functions_special as f_sp
from libc.stdio cimport printf
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport pow,sqrt,exp,log,fabs,M_PI

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

def avg_ml_poly_r( np.ndarray[float,ndim=1,mode="c"] R, int Ns, float f, int m, int M ):
   cdef:
      int i,NPOINTS
      double s,AVG_ML,Z,Lim,eps
   eps           = 1e-15
   NPOINTS       = 1000
   AVG_ML        = 0.0
   if m == 1 :
      s          = 0.0
      for i in range(0,Ns):
         s       = s + ( -f * log( 1. + R[ i ] ) )
      if f != 1. :
         Z       = ( pow(2.,1.-f) - pow(1.,1.-f) ) / ( 1. - f )
      else :
         Z       = log( 2.0 )
      s          = s / float(Ns)
      AVG_ML     = s - log( Z )
   else :
      Z          = 0.
      s          = 0.0
      for i in range(0,Ns):
         Lim     = 0
         for j in range(1,M+1):
            Lim  = Lim + ( pow( -R[ i], np.double(j) ) / pow( np.double(j), np.double(m) ) )
         s       = s + ( f * Lim )
      r_do       = np.asarray( np.arange(0,1.0 + float(1. / float(NPOINTS)), float(1. / float(NPOINTS)) ) , dtype=np.float32)
      r_do[0]    = eps
      r_do[NPOINTS]= 1.0 - eps
      Z          = 0.0
      for i in range(0,NPOINTS):
         Lim     = 0
         for j in range(1,M+1):
            Lim  = Lim + ( pow( -r_do[ i], np.double(j) ) / pow( np.double(j), np.double(m) ) )
         Z       = Z + exp( f * Lim )
      Z          = Z * ( 1. / np.double(NPOINTS) ) # multiply by limiting dr for integration
      s          = s / float(Ns)
      AVG_ML     = s - log( Z )
   return AVG_ML

def avg_ml_sg_r( np.ndarray[float,ndim=1,mode="c"] R, int Ns, float f, float tau ):
   cdef:
      int i
      double s,AVG_ML,Z,Ei_1,Ei_0,x_1
   
   AVG_ML     = 0.0
   s          = 0.0
   for i in range(0,Ns):
      s       = s + ( f*( (1. / (1. + (tau * R[ i ]) )) - 1. ) )
   
   x_1        = f / (1. + (tau ))
   Ei_1       = f_sp.Ei( x_1 )
   Ei_0       = f_sp.Ei( f )
   Z          = ((1. + tau) / tau)*exp( x_1 - f ) - ((f*exp(-f) / tau) * Ei_1)
   Z          = Z - ( (1. / tau) - ((f*exp(-f) / tau) * Ei_0) )
   s          = s / float(Ns)
   #print("f={}, tau={}, s = {}, Z = {}, Ei0 = {}, Ei1 = {}".format(f,tau,s,Z,Ei_0,Ei_1))
   AVG_ML     = s - log( Z )
   return AVG_ML

def avg_ml_fo_r( np.ndarray[float,ndim=1,mode="c"] R, int Ns, float f ):
   cdef:
      int i
      double s,AVG_ML,Z
   
   AVG_ML     = 0.0
   s          = 0.0
   for i in range(0,Ns):
      s       = s + ( -f* R[ i ] )
   Z          = (1.0 - exp(-f)) / f
   s          = s / float(Ns)
   AVG_ML     = s - log( Z )
   return AVG_ML

def avg_ml_so_r( np.ndarray[float,ndim=1,mode="c"] R, int Ns, float f1, float f2 ):
   cdef:
      int i
      double s1,s2,AVG_ML,Z,K,erfi_0,erfi_1
   
   AVG_ML     = 0.0
   s1         = 0.0
   for i in range(0,Ns):
      s1      = s1 + ( -(f1 * R[ i ]) + (f2 * R[i] * R[i] )  )
   
   s2         = f1 / ( 2.0 * sqrt(f2) )
   K          = ( sqrt( M_PI ) / (2.0 * sqrt(f2)) ) * exp( -(s2*s2) )
   erfi_0     = f_sp.erfi( s2 )
   erfi_1     = f_sp.erfi( s2 - sqrt(f2) )
   Z          = K * ( erfi_0 - erfi_1 )
   
   s1         = s1 / float(Ns)
   AVG_ML     = s1 - log( Z )
   return AVG_ML

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the polylogarithmic exponential distr. model to a given dataset of population
# rates (number of active neurons divided by the population size), assummed to come from
# spontaneous neural activity.
# Parameters:
# ---R        : float one-dimensional array with Ns population rate samples from spontaneous
#               data, with population size N.
# ---r_dom    : float one-dimensional array with the domain values for the population rate
#               variable (points 0/N, 1/N, 2/N, ..., N/N).
# ---N        : integer value with the size of the neural population.
# ---Ns       : integer value with the number of samples.
# ---M_TERMS  : integer value with the number of terms to consider in the alternating series
#               for the case of m > 1. This value should be in [3,4,5,...,N].
# ---eta      : float value with the learning rate for locally optimizing over the f parameter.
# ---f_init   : float value with the initial value for the sparsity inducing parameter.
# ---m        : integer value for the integer order of the polylogarithmic function.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Fitted parameter: f, where f is the sparsity inducing parameter.
@boundscheck(False)
@wraparound(False)
def model_fit_polylogarithmic_r(  np.ndarray[float,ndim=1,mode="c"] R, \
                                  np.ndarray[float,ndim=1,mode="c"] r_dom, int N, int Ns, int M_TERMS,
                                  float eta, float f_init, int m, int MAX_ITE, int T ):
   cdef:
      int ite,t,i
      float f,s1,s2,eps,conv_thresh
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_wrt_f     = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = pdf_r.copy(order='C')
   eps                            = 0.00001
   conv_thresh                    = 0.5
   # Initialize parameters
   f                              = f_init
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      # Locally optimize for the sparsity inducing parameter f given m
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,N+1):
            lZ_der_wrt_f[i] = 0.0
         # Computation of first part of log-likelihood derivative
         for i in range(0,Ns):
            c_der_ll_poly_r_wrt_f_part( i, &ll_batch[0], M_TERMS, m, &R[0] )
         # Collapse batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         
         # Computation of second part of log-likelihood derivative
         for i in range(0,N+1):
            c_der_ll_poly_r_wrt_f_part( i, &lZ_der_wrt_f[0], M_TERMS,m, &r_dom[0] )
         
         for i in range(0,N+1):
            pdf_r[i] = 0.0
         f_gen.polylogarithmic_pdf( pdf_r, r_dom, N+1, f, m, M_TERMS )
         # Collapse batch results
         s2  = 0.0
         for i in range(0,N+1):
           s2= s2 + ( pdf_r[ i ] * lZ_der_wrt_f[ i ])
         f   = f + (eta*(s1-s2))
         if np.abs( s1 - s2 ) <= conv_thresh :
            print("   Small gradient reached")
            ite = MAX_ITE
            break
         if f <= 0.0 :
            f = eps #0.0
      print("Last gradient magnitude = {}".format( np.abs( s1 - s2 ) ))
      eta    = eta / (1.0 + (0.0001)*float(ite))
      print("model_fit_polylogarithmic_r:: iteration {}, f={}, m={}".format(ite,f,m))
      if ite == MAX_ITE :
         break
   return f

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the shifted-geometric exponential distr. model to a given dataset of population
# rates (number of active neurons divided by the population size), assummed to come from
# spontaneous neural activity.
# Parameters:
# ---R        : float one-dimensional array with Ns population rate samples from spontaneous
#               data, with population size N.
# ---r_dom    : float one-dimensional array with the domain values for the population rate
#               variable (points 0/N, 1/N, 2/N, ..., N/N).
# ---N        : integer value with the size of the neural population in each sample.
# ---Ns       : integer value with the number of samples in the binary data.
# ---eta      : float value with the learning rate for locally optimizing the f-parameter.
# ---f_init   : float value with the initial value for the sparsity inducing parameter.
# ---tau      : float value with the shifted-geometric tau parameter.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Fitted parameter: f, where f is the sparsity inducing parameter.
@boundscheck(False)
@wraparound(False)
def model_fit_shifted_geom_r( np.ndarray[float,ndim=1,mode="c"] R, \
                              np.ndarray[float,ndim=1,mode="c"] r_dom, int N, int Ns, float eta,\
                              float f_init, float tau, int MAX_ITE, int T ):
   cdef:
      int ite,t,i
      float f,s1,s2,eps,mag,conv_thresh
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_batch     = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = pdf_r.copy(order='C')
   eps                            = 0.00001
   conv_thresh                    = 0.5
   # Initialize parameters
   f         = f_init
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      # Locally optimize for the sparsity inducing parameter f given tau
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,N+1):
            lZ_der_batch[i] = 0.0
         # Computation of first part of log-likelihood derivative
         for i in range(0,Ns):
            c_der_ll_sg_r_wrt_f_part( i, &ll_batch[0], tau, &R[0] )
         # Collapse batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         
         # Computation of second part of log-likelihood derivative
         for i in range(0,N+1):
            c_der_ll_sg_r_wrt_f_part( i, &lZ_der_batch[0], tau, &r_dom[0] )
         
         for i in range(0,N+1):
            pdf_r[i] = 0.0
         f_gen.shifted_geometric_pdf( pdf_r, r_dom, N+1, f, tau)
         # Collapse batch results
         s2  = 0.0
         for i in range(0,N+1):
           s2= s2 + ( pdf_r[ i ] * lZ_der_batch[ i ])
         
         f   = f + (eta*(s1-s2))
         if np.abs( s1 - s2 ) <= conv_thresh :
            print("   Small gradient reached")
            ite = MAX_ITE
            break
         if f <= 0.0 :
            f = eps #0.0
      print("Last gradient magnitude = {}".format( np.abs( s1 - s2 ) ))
      eta    = eta / (1.0 + (0.0001)*float(ite))
      print("model_fit_shifted_geom_r:: iteration {}, f={}, tau={}".format(ite,f,tau))
      if ite == MAX_ITE :
         break
   return f

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the first-order interactions exponential distr. model to a given dataset of
# population rates (number of active neurons divided by the population size), assummed to come
# from spontaneous neural activity.
# Parameters:
# ---R        : float one-dimensional array with Ns population rate samples from spontaneous
#               data, with population size N.
# ---r_dom    : float one-dimensional array with the domain values for the population rate
#               variable (points 0/N, 1/N, 2/N, ..., N/N).
# ---N        : integer value with the size of the neural population in each sample.
# ---Ns       : integer value with the number of samples in the binary data.
# ---eta      : float value with the learning rate for optimizing the f-parameter.
# ---f_init   : float value with the initial value for the sparsity inducing parameter.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Fitted parameter: f, the sparsity inducing parameter.
@boundscheck(False)
@wraparound(False)
def model_fit_first_ord_r( np.ndarray[float,ndim=1,mode="c"] R, \
                           np.ndarray[float,ndim=1,mode="c"] r_dom, int N, int Ns, \
                           float eta, float f_init, int MAX_ITE, int T ):
   cdef:
      int ite,t,i
      float f,s1,s2,eps,conv_thresh
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_batch     = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = pdf_r.copy(order='C')
   eps                            = 0.00001
   conv_thresh                    = 0.5
   # Initialize parameters
   f         = f_init
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      # Locally optimize for the sparsity inducing parameter f
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,N+1):
            lZ_der_batch[i] = 0.0
         # Computation of first part of log-likelihood derivative
         for i in range(0,Ns):
            c_der_ll_first_o_r_wrt_f_part( i, &ll_batch[0], &R[0] )
         # Collapse batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         
         # Computation of second part of log-likelihood derivative
         for i in range(0,N+1):
            c_der_ll_first_o_r_wrt_f_part( i, &lZ_der_batch[0], &r_dom[0] )
         
         for i in range(0,N+1):
            pdf_r[i] = 0.0
         f_gen.first_ord_pdf( pdf_r, r_dom, N+1, f )
         # Collapse batch results
         s2  = 0.0
         for i in range(0,N+1):
           s2= s2 + ( pdf_r[ i ] * lZ_der_batch[ i ])
         
         f   = f + (eta*(s1-s2))
         if np.abs( s1 - s2 ) <= conv_thresh :
            print("   Small gradient reached")
            ite = MAX_ITE
            break
         if f <= 0.0 :
            f = eps #0.0
      print("Last gradient magnitude = {}".format( np.abs( s1 - s2 ) ))
      eta    = eta / (1.0 + (0.0001)*float(ite))
      print("model_fit_first_ord_r:: iteration {}, f={}".format(ite,f))
      if ite == MAX_ITE :
         break
   return f

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the second-order interactions exponential distr. model to a given dataset of
# population rates (number of active neurons divided by the population size), assummed to come
# from spontaneous neural activity.
# Parameters:
# ---R        : float one-dimensional array with Ns population rate samples from spontaneous
#               data, with population size N.
# ---r_dom    : float one-dimensional array with the domain values for the population rate
#               variable (points 0/N, 1/N, 2/N, ..., N/N).
# ---N        : integer value with the size of the neural population in each sample.
# ---Ns       : integer value with the number of samples in the binary data.
# ---eta1      : float value with the learning rate for optimizing the f1-parameter.
# ---eta2      : float value with the learning rate for optimizing the f2-parameter.
# ---f_init   : float value with the initial value for parameters.
# ---MAX_ITE  : integer value with the maximum number of optimization iterations.
# ---T        : integer value with the maximum number of local optimization iterations.
# Returns:
# ---Fitted parameters: f1 and f2.
@boundscheck(False)
@wraparound(False)
def model_fit_second_ord_r( np.ndarray[float,ndim=1,mode="c"] R, \
                           np.ndarray[float,ndim=1,mode="c"] r_dom, int N, int Ns, \
                           float eta1, float eta2, float f_init, int MAX_ITE, int T ):
   cdef:
      int ite,t,i,converged_f1
      float f1,f2,s1,s2,eps,conv_thresh
   cdef float[:] ll_batch         = np.zeros((Ns,),dtype=np.float32)
   cdef float[:] lZ_der_batch     = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = np.zeros((N+1,),dtype=np.float32)
   pdf_r                          = pdf_r.copy(order='C')
   eps                            = 0.00001
   conv_thresh                    = 0.5
   # Initialize parameters
   f1        = f_init
   f2        = f_init
   # Iterative optimization
   for ite in range(1,MAX_ITE+1):
      converged_f1 = 0
      # Locally optimize for the sparsity inducing parameter f1
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,N+1):
            lZ_der_batch[i] = 0.0
         # Computation of first part of log-likelihood derivative
         for i in range(0,Ns):
            c_der_ll_first_o_r_wrt_f_part( i, &ll_batch[0], &R[0] )
         # Collapse batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         
         # Computation of second part of log-likelihood derivative
         for i in range(0,N+1):
            c_der_ll_first_o_r_wrt_f_part( i, &lZ_der_batch[0], &r_dom[0] )
         
         for i in range(0,N+1):
            pdf_r[i] = 0.0
         f_gen.second_ord_pdf( pdf_r, r_dom, N+1, f1,f2 )
         # Collapse batch results
         s2  = 0.0
         for i in range(0,N+1):
           s2= s2 + ( pdf_r[ i ] * lZ_der_batch[ i ])
         
         f1  = f1 + (eta1*(s1-s2))
         if np.abs( s1 - s2 ) <= conv_thresh :
            print("   Small gradient reached")
            converged_f1 = 1
            break
         if f1 <= 0.0 :
            f1 = eps #0.0
      print("Last gradient magnitude = {}".format( np.abs( s1 - s2 ) ))
      # Locally optimize for the second-order parameter f2
      # ---------------------------------------------------------------------------
      for t in range(1,T+1):
         for i in range(0,Ns):
            ll_batch[i]     = 0.0
         for i in range(0,N+1):
            lZ_der_batch[i] = 0.0
         # Computation of first part of log-likelihood derivative
         for i in range(0,Ns):
            c_der_ll_second_o_r_wrt_f2_part( i, &ll_batch[0], &R[0] )
         # Collapse batch results
         s1  = 0.0
         for i in range(0,Ns):
           s1= s1 + ll_batch[ i ]
         s1  = s1 / float( Ns )
         
         # Computation of second part of log-likelihood derivative
         for i in range(0,N+1):
            c_der_ll_second_o_r_wrt_f2_part( i, &lZ_der_batch[0], &r_dom[0] )
         
         for i in range(0,N+1):
            pdf_r[i] = 0.0
         f_gen.second_ord_pdf( pdf_r, r_dom, N+1, f1,f2 )
         # Collapse batch results
         s2  = 0.0
         for i in range(0,N+1):
           s2= s2 + ( pdf_r[ i ] * lZ_der_batch[ i ] )
         f2  = f2 + (eta2*(s1-s2))
         if np.abs( s1 - s2 ) <= conv_thresh :
            print("   Small gradient reached")
            if converged_f1 == 1 :
               ite = MAX_ITE
            break
         if f2 <= 0 :
            f2 = eps #0.
      eta1   = eta1 / (1.0 + (0.0001)*float(ite))
      eta2   = eta2 / (1.0 + (0.0001)*float(ite))
      print("Last gradient magnitude = {}".format( np.abs( s1 - s2 ) ))
      print("model_fit_second_ord_r:: iteration {}, f1={}, f2={}".format(ite,f1,f2))
      if ite == MAX_ITE :
         break
   return f1,f2

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Declaration of all c code functions called from within this Cython file.
# The header function declaration file must be located in the extern from
# "path_to_c_header_file.h" defined below.
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
cdef extern from "c/c_functions_numerical.h" nogil:
   
   void c_der_ll_poly_r_wrt_f_part( int i, float *ll_batch, int M_terms, int m, float *R )
   void c_log_likelihood_poly_r_part_i( int i, float *R, float *ll_batch, int M_terms, float f, int m )
   void c_log_likelihood_sg_r_part_i( int i, float *R, float *ll_batch, float f, float tau )
   void c_der_ll_sg_r_wrt_f_part( int i, float *ll_batch, float tau, float *R )
   void c_der_ll_sg_r_wrt_tau_part( int i, float *ll_batch, float f, float tau, float *R )
   void c_der_ll_first_o_r_wrt_f_part( int i, float *ll_batch, float *R )
   void c_der_ll_second_o_r_wrt_f2_part( int i, float *ll_batch, float *R )