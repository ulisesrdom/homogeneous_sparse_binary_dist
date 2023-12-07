// Declaration of all the c code functions for the computations involved
// at the learning stage of the Bayesian model.
// Each function is described in the file "c_functions_sampling.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
void c_initRandom();
void c_polylogarithmic_samp( int r, int NITE, int *X, int N, float F );
void c_shifted_geometric_samp( int r, int NITE, int *X, int N, float F, float tau );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------