"# homogeneous_sparse_binary_dist" 
Example execution for fitting the shifted-geometric exp. distribution to spiking data:

python main.py -OPT 4 -SAMP_T 0 -VIS_C_T 0 -DIST_T 2 -N_SAMP 15000 -N_P 100 -N 1000 -BASE_PAR 100.,0.7,10,1.0,0.01,5,20 -IN_F ..\DATA -IN_FI spont_M160825_MP027_2016-12-12.p,spont_M160907_MP028_2016-09-26.p -OUT_F ..\RESULTS\ -COLOR_L red,green -LSTY dotted,dashed -DPI 500
