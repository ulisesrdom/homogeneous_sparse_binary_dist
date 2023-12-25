// Declaration of the c code functions where numerical procedures are involved.
// Each function is described in the file "c_functions_numerical.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
void c_der_ll_poly_wrt_f_p1( int r, float *ll_batch, int m, int *X, int N, int Ns );
void c_der_Z_poly_wrt_f( int r, float *lZ_der_wrt_f, int m, int *X, int N, int Ns );
void c_log_likelihood_poly_r( int r, int *X, float *ll_batch, float f, float m,  int N );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------