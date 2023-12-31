// Declaration of the c code functions where numerical procedures are involved.
// Each function is described in the file "c_functions_numerical.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
void c_der_ll_poly_wrt_f_part( int i, float *ll_batch, int m, int *X, int N );
void c_log_likelihood_poly_part_i( int i, int *X, float *ll_batch, float f, float m,  int N );
void c_der_ll_sg_wrt_f_part( int i, float *ll_batch, float tau, int *X, int N );
void c_der_ll_sg_wrt_tau_part( int i, float *ll_batch, float f, float tau, float *X, int N );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------