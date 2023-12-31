#include "c_functions_sampling.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------------------------
// Initialize the seed for random number generation.
// Inputs :
//    void (no input).
// Outputs:
//    void (no return value).
// -----------------------------------------------------------------------------------------------
void c_initRandom(){
   srand( 111 );
}

// -----------------------------------------------------------------------------------------------
// Compute individual conditional probabilities of activation for each neuron k
// and simulate the corresponding sample values for each binary neuron based on the
// Gibbs sampling scheme for the binary polylogarithmic model. That is
// P( x_k = 1 | x_{-k}, theta_N )  for each k and corresponding simulation of x_k for each k.
// Inputs :
//    r       (integer value with the index that identifies a Gibbs sample stored in the
//             array X; the N-dimensional r-th sample in X is indexed as
//             X[ r * N ], X[ r * N + 1 ], ... , X[ r * N + (N-1)]).
//    NITE    (integer with number of simulation loops).
//    X       (pointer to integer array where the simulated binary values will be stored for
//             each neuron; X[ r * N  +  k ] stores the binary value of neuron k (out of N
//             neurons in the population) for the r-th Gibbs sample).
//    N       (integer value with the size of the population of neurons).
//    M_TERMS (integer value with the number of terms to consider in the alternating series
//             for the case of m > 1. This value should be in [3,4,5,...,N].).
//    F       (float value with the f sparsity-inducing parameter).
//    m       (integer value with the order of the polylogarithmic function).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_polylogarithmic_samp( int r, int NITE, int *X, int N, int M_TERMS, float F, int m ){
   int k,j,samp,ite,si;
   float P,q,term0,term1,f_term,log_sum,sum_over_x;
   if( m ==1 ){
      for( ite = 1 ; ite <= NITE ; ite++ ){
	    sum_over_x  = 0.0;
        samp        = r * N ;
        for( j=0 ; j < N ; j++ ){
           sum_over_x = sum_over_x + ((float)X[ samp + j ]);
        }
        for( k=0; k < N ; k++ ){
           samp      = r * N ;
	       
           // Compute  F [ (term|x_k=0) - (term|x_k=1)]
		   // --------------------------------------------------------------------------------
		   term0 = (sum_over_x - ((float)X[ samp + k ])) / ( (float) N ) ;
	       if( fabs( term0 ) + 1.0 <= 1e-08 ){
		       printf("::c_polylogarithm_samp:: DIVISION BY ZERO term0...");
		       return;
	       }
	       term0 = log( term0 + 1.0  );
	       
	       term1 = (1.0 + (sum_over_x - ((float)X[ samp + k ])) ) / ( (float) N ) ;
	       if( fabs( term1 ) + 1.0 <= 1e-08 ){
		       printf("::c_polylogarithm_samp:: DIVISION BY ZERO term1...");
		       return;
	       }
	       term1 = log( term1 + 1.0 );
		   
		   f_term= F * (term0 - term1);
		   
		   // Compute log(h( sum_over_x_not_k + 1) / h(sum_over_x_not_k + 0))
		   // --------------------------------------------------------------------------------
		   log_sum   = log( (1.0 + sum_over_x - ((float)X[ samp + k ])) / ( ((float)N) - (sum_over_x - ((float)X[ samp + k ])) ) ) ;
		   
		   // Compute conditional probability for sampling
		   // --------------------------------------------------------------------------------
		   P         = 1.0 / (  1.0 + exp( -( log_sum + f_term ) )  );
		   
           q         = ( (float)rand() )  /  ( (float)RAND_MAX );
           
           if( q < P ){
		      sum_over_x    = sum_over_x + ((float)(1 - X[ samp + k ]));
              X[ samp + k ] = 1;
           }else{
		      sum_over_x    = sum_over_x + ((float)(0 - X[ samp + k ]));
              X[ samp + k ] = 0;
           }
	    }
      }
   }else{
      if( m < 1 ){
         printf("::c_polylogarithm_samp:: wrong value for m...");
         return;
	  }else{
         for( ite = 1 ; ite <= NITE ; ite++ ){
	       sum_over_x  = 0.0;
           samp        = r * N ;
           for( j=0 ; j < N ; j++ ){
              sum_over_x = sum_over_x + ((float)X[ samp + j ]);
           }
           for( k=0; k < N ; k++ ){
              samp      = r * N ;
	          
      		  // Compute  alternating series part
		      // --------------------------------------------------------------------------------
			  si        = -1 ;
			  f_term    = 0.0;
		      for( j = 1; j <= M_TERMS ; j++ ){
				  si    = -1 * si ;
				  term0 = pow( ( sum_over_x - ((float)X[ samp + k ]) ) / ((float)N) , (float)k );
				  term1 = pow( ( 1.0 + sum_over_x - ((float)X[ samp + k ]) ) / ((float)N) , (float)k );
				  f_term= f_term + ( ((float)si) * (term0 - term1) / pow( (float)j, (float)m ) );
			  }
		      
		      f_term= F * f_term ;
		      
		      // Compute log(h( sum_over_x_not_k + 1) / h(sum_over_x_not_k + 0))
		      // --------------------------------------------------------------------------------
		      log_sum   = log( (1.0 + sum_over_x - ((float)X[ samp + k ])) / ( ((float)N) - (sum_over_x - ((float)X[ samp + k ])) ) ) ;
		      
		      // Compute conditional probability for sampling
		      // --------------------------------------------------------------------------------
		      P         = 1.0 / (  1.0 + exp( -( log_sum + f_term ) )  );
		      
              q         = ( (float)rand() )  /  ( (float)RAND_MAX );
              
              if( q < P ){
		         sum_over_x    = sum_over_x + ((float)(1 - X[ samp + k ]));
                 X[ samp + k ] = 1;
              }else{
		         sum_over_x    = sum_over_x + ((float)(0 - X[ samp + k ]));
                 X[ samp + k ] = 0;
              }
	       }
         }
	  }
   }
}

// -----------------------------------------------------------------------------------------------
// Compute individual conditional probabilities of activation for each neuron k
// and simulate the corresponding sample values for each binary neuron based on the
// Gibbs sampling scheme for the binary shifted-geometric model. That is
// P( x_k = 1 | x_{-k}, theta_N )  for each k and corresponding simulation of x_k for each k.
// Inputs :
//    r       (integer value with the index that identifies a Gibbs sample stored in the
//             array X; the N-dimensional r-th sample in X is indexed as
//             X[ r * N ], X[ r * N + 1 ], ... , X[ r * N + (N-1)]).
//    NITE    (integer with number of simulation loops).
//    X       (pointer to integer array where the simulated binary values will be stored for
//             each neuron; X[ r * N  +  k ] stores the binary value of neuron k (out of N
//             neurons in the population) for the r-th Gibbs sample).
//    N       (integer value with the size of the population of neurons).
//    F       (float value with the f sparsity-inducing parameter).
//    tau     (float value with the tau parameter specific for the shifted-geometric model).
// Outputs:
//    void    (no return value).
// -----------------------------------------------------------------------------------------------
void c_shifted_geometric_samp( int r, int NITE, int *X, int N, float F, float tau ){
   int k,j,samp,ite;
   float P,q,shift_geom_1,shift_geom_0,f_term,log_sum,sum_over_x;
   for( ite = 1 ; ite <= NITE ; ite++ ){
	 
	 sum_over_x  = 0.0;
	 samp        = r * N ;
     for( j=0 ; j < N ; j++ ){
        sum_over_x = sum_over_x + ((float)X[ samp + j ]);
     }
     for( k=0; k < N ; k++ ){
        samp      = r * N ;
	    
		// Compute  F [ (shifted-geometric|x_k=1) - (shifted-geometric|x_k=0)]
		// --------------------------------------------------------------------------------
		shift_geom_1 = 1. / ( 1.0 +  tau * (1.0 + (sum_over_x - ((float)X[ samp + k ])) ) / ( (float) N )) ;
		
		shift_geom_0 = 1. / ( 1.0 +  tau * ( (sum_over_x - ((float)X[ samp + k ])) ) / ( (float) N )) ;
		
		f_term    = F * (shift_geom_1 - shift_geom_0);
		
		// Compute log(h( sum_over_x_not_k + 1) / h(sum_over_x_not_k + 0))
		// --------------------------------------------------------------------------------
		log_sum   = log( (1.0 + sum_over_x - ((float)X[ samp + k ])) / ( ((float)N) - (sum_over_x - ((float)X[ samp + k ])) ) ) ;
		
		// Compute conditional probability for sampling
		// --------------------------------------------------------------------------------
		P         = 1.0 / (  1.0 + exp( -( log_sum + f_term ) )  );
		
        q         = ( (float)rand() )  /  ( (float)RAND_MAX );
        
        if( q < P ){
		   sum_over_x    = sum_over_x + ((float)(1 - X[ samp + k ]));
           X[ samp + k ] = 1;
        }else{
		   sum_over_x    = sum_over_x + ((float)(0 - X[ samp + k ]));
           X[ samp + k ] = 0;
        }
	 }
   }
}