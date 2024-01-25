#include "c_functions_numerical.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------------------------
// Compute part of the derivative of the log-likelihood function with respect to the sparsity-
// inducing parameter for the polylogarithmic exp. distr. case for the population rate variable.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in R, where 
//             Ns is the number of samples).
//    ll_batch (pointer to float array where the results of the derivative are to be stored for
//            each sample).
//    M_terms (integer value with the number of terms to use for the polylogarithmic series
//             approximation).
//    m       (integer value with the order of the polylogarithmic function).
//    R       (pointer to float array where each pop. rate sample is stored; R[ i ]
//             stores the number of active neurons divided by N (the population size) for the
//             i-th sample).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_ll_poly_r_wrt_f_part( int i, float *ll_batch, int M_terms, int m, float *R ){
   int k;
   float s;
   s           = 0.0 ;
   if( m == 1 ){
      s        = -log( 1.0 + R[ i ] ) ;
   }else{
      for( k = 1; k <= M_terms ; k++ ){
         s     = s + ( pow( -R[ i ] , (float)k ) / pow( (float)k, (float)m ) ) ;
      }
   }
   ll_batch[ i ] = s ;
}

// -----------------------------------------------------------------------------------------------
// Compute the log-likelihood partial function value (where the normalization function values are
// ommitted) for a single population rate sample i for the polylogarithmic exp. distr and store it
// in its corresponding cell position in a batch vector.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in R, where 
//             Ns is the number of samples).
//    R       (pointer to float array where each pop. rate sample is stored; R[ i ]
//             stores the number of active neurons divided by N (the population size) for the
//             i-th sample).
//    ll_batch (pointer to float array where the results of the function are to be stored for
//            each sample).
//    M_terms (integer value with the number of terms to use for the polylogarithmic series
//             approximation).
//    f       (float value with the sparsity-inducing parameter).
//    m       (integer value with the order of the polylogarithmic function).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_log_likelihood_poly_r_part_i( int i, float *R, float *ll_batch, int M_terms, float f, float m ){
   int k;
   float s;
   s           = 0.0 ;
   if( m == 1 ){
      s        = -log( 1.0 + R[ i ] ) ;
   }else{
      for( k = 1; k <= M_terms ; k++ ){
         s     = s + ( pow( -R[ i ] , (float)k ) / pow( (float)k, (float)m ) ) ;
      }
   }
   ll_batch[ i ] = f*s ;
}

// -----------------------------------------------------------------------------------------------
// Compute part of the derivative of the log-likelihood function with respect to the sparsity-
// inducing parameter for the shifted-geometric exp. distr. case for the population rate variable.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in R, where 
//             Ns is the number of samples).
//    ll_batch (pointer to float array where the results of the function are to be stored for
//            each sample).
//    tau     (float value with the shifted-geometric tau parameter).
//    R       (pointer to float array where each pop. rate sample is stored; R[ i ]
//             stores the number of active neurons divided by N (the population size) for the
//             i-th sample).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_ll_sg_r_wrt_f_part( int i, float *ll_batch, float tau, float *R ){
   float s;
   s             = ( 1.0 / ( 1.0 + ( tau*R[i] ) ) ) - 1.0;
   ll_batch[ i ] = s ;
}

// -----------------------------------------------------------------------------------------------
// Compute part of the derivative of the log-likelihood function with respect to the shifted-
// geometric parameter tau for the shifted-geometric exp. distr. case for the population rate variable.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in R, where 
//             Ns is the number of samples).
//    ll_batch (pointer to float array where the results of the function are to be stored for
//            each sample).
//    f       (float value with the sparsity-inducing parameter).
//    tau     (float value with the shifted-geometric tau parameter).
//    R       (pointer to float array where each pop. rate sample is stored; R[ i ]
//             stores the number of active neurons divided by N (the population size) for the
//             i-th sample).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_ll_sg_r_wrt_tau_part( int i, float *ll_batch, float f, float tau, float *R ){
   float s;
   s             = ( R[i] / pow( 1.0 + ( tau*R[i] ) , 2.0 )  ) ;
   ll_batch[ i ] = -f * s ;
}