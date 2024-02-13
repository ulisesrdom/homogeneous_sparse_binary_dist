// Declaration of the c code functions where numerical procedures are involved.
// Each function is described in the file "c_functions_numerical.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
void c_der_ll_poly_r_wrt_f_part( int i, float *ll_batch, int M_terms, int m, float *R );
void c_log_likelihood_poly_r_part_i( int i, float *R, float *ll_batch, int M_terms, float f, int m );
void c_log_likelihood_sg_r_part_i( int i, float *R, float *ll_batch, float f, float tau );
void c_der_ll_sg_r_wrt_f_part( int i, float *ll_batch, float tau, float *R );
void c_der_ll_sg_r_wrt_tau_part( int i, float *ll_batch, float f, float tau, float *R );
void c_der_ll_first_o_r_wrt_f_part( int i, float *ll_batch, float *R );
void c_der_ll_second_o_r_wrt_f2_part( int i, float *ll_batch, float *R );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------