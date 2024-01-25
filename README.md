"# homogeneous_sparse_binary_dist" 
Example execution for fitting both the polylogarithmic and the shifted-geometric exp. distributions to spiking data:

python main.py -OPT 4 -SAMP_T 0 -VIS_C_T 0 -DIST_T 3 -N_SAMP 50000 -N_P 50 -N 126 -BASE_PAR 5.0,1,3,0.5,20,0.02,0.000002,100,20 -IN_F ..\DATA\ALLEN\ -IN_FI V1_spontaneous.p -OUT_F ..\RESULTS\ -COLOR_L green,red -LSTY None,dashed -DPI 500
