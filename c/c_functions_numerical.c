#include "c_functions_numerical.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


// -----------------------------------------------------------------------------------------------
// Compute the first part of the derivative of the log-likelihood function with respect to
// the sparsity-inducing parameter.
// Inputs :
//    i       (integer value with the index in 0 to Ns-1 that identifies a sample in X).
//    ll_batch (pointer to float array where the results of the derivative are to be stored for
//            each sample).
//    m       (integer value with the order of the polylogarithmic function).
//    X       (pointer to integer array where each binary vector sample is stored; X[ r * N  +  k ]
//             stores the binary value of neuron k (out of N neurons in the population) for the
//             r-th sample).
//    N       (integer value with the size of the population of neurons).
//    Ns      (integer value with the number of samples).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_ll_poly_wrt_f_p1( int i, float *ll_batch, int m, int *X, int N, int Ns ){
   int k;
   float sum_x,s,si;
   sum_x       = 0.0 ;
   for( k = 0; k < N ; k++ ){
	   sum_x   = sum_x + ((float)X[ (i*N) + k ]);
   }
   s           = 0.0 ;
   si          = -1.0;
   for( k = 0; k < N ; k++ ){
	   si      = -1.0 * si ;
	   s       = s + ( si * pow( sum_x , (float)(k+1) ) / pow( (float)(k+1), (float)m ) );
   }
   ll_batch[ i ] = s ;
}

// -----------------------------------------------------------------------------------------------
// Compute the derivative of the log-partition function with respect to
// the sparsity-inducing parameter.
// Inputs :
//    r       (integer value with the index in 0 to M-1 that identifies a sample in X).
//    ll_batch (pointer to float array where the results of the derivative are to be stored for
//            each sample).
//    m       (integer value with the order of the polylogarithmic function).
//    X       (pointer to integer array where each binary vector sample is stored; X[ r * N  +  k ]
//             stores the binary value of neuron k (out of N neurons in the population) for the
//             r-th sample).
//    N       (integer value with the size of the population of neurons).
//    Ns      (integer value with the number of samples).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_der_Z_poly_wrt_f( int r, float *lZ_der_wrt_f, int m, int *X, int N, int Ns ){
   
}
void c_log_likelihood_poly_r( int r, int *X, float *ll_batch, float f, float m,  int N ){
   
}