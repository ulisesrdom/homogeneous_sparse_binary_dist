// Declaration of the c code sampling functions.
// Each function is described in the file "c_functions_sampling.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
void c_initRandom();
void c_polylogarithmic_samp( int r, int NITE, int *X, int N, int M_TERMS, float F, int m );
void c_shifted_geometric_samp( int r, int NITE, int *X, int N, float F, float tau );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------