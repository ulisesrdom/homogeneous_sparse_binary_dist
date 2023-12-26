#include "c_functions_numerical.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


// -----------------------------------------------------------------------------------------------
// Compute part of the derivative of the log-likelihood function with respect to the sparsity-
// inducing parameter.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in X, where 
//             Ns is the number of samples).
//    ll_batch (pointer to float array where the results of the derivative are to be stored for
//            each sample).
//    m       (integer value with the order of the polylogarithmic function).
//    X       (pointer to integer array where each binary vector sample is stored; X[ i * N  +  k ]
//             stores the binary value of neuron k (out of N neurons in the population) for the
//             i-th sample).
//    N       (integer value with the size of the population of neurons).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_ll_poly_wrt_f_part( int i, float *ll_batch, int m, int *X, int N ){
   int k;
   float sum_x_N,s,si;
   sum_x_N     = 0.0 ;
   for( k = 0; k < N ; k++ ){
	   sum_x_N = sum_x_N + ((float)X[ (i*N) + k ]);
   }
   sum_x_N     = sum_x_N / ((float)N);
   s           = 0.0 ;
   si          = -1.0;
   for( k = 1; k <= N ; k++ ){
	   si      = -1.0 * si ;
	   s       = s + ( si * pow( sum_x_N , (float)k ) / pow( (float)k, (float)m ) );
   }
   ll_batch[ i ] = s ;
}

// -----------------------------------------------------------------------------------------------
// Compute the log-likelihood partial function value (where the base measure function and the 
// normalization function values are ommitted) for a single binary vector sample i for the
// polylogarithmic exp. distr and store it in its corresponding cell position in a batch vector.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in X, where 
//             Ns is the number of samples).
//    X       (pointer to integer array where each binary vector sample is stored; X[ i * N  +  k ]
//             stores the binary value of neuron k (out of N neurons in the population) for the
//             i-th sample).
//    ll_batch (pointer to float array where the results of the function are to be stored for
//            each sample).
//    f       (float value with the sparsity-inducing parameter).
//    m       (integer value with the order of the polylogarithmic function).
//    N       (integer value with the size of the population of neurons).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_log_likelihood_poly_part_i( int i, int *X, float *ll_batch, float f, float m,  int N ){
   int k;
   float sum_x_N,s,si;
   sum_x_N     = 0.0 ;
   for( k = 0; k < N ; k++ ){
	   sum_x_N = sum_x_N + ((float)X[ (i*N) + k ]);
   }
   sum_x_N     = sum_x_N / ((float)N);
   s           = 0.0 ;
   si          = -1.0;
   for( k = 1; k <= N ; k++ ){
	   si      = -1.0 * si ;
	   s       = s + ( si * pow( sum_x_N , (float)k ) / pow( (float)k, (float)m ) );
   }
   ll_batch[ i ] = -f*s ;
}