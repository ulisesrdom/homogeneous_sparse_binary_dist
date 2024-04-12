# -*- coding: utf-8 -*-
import numpy as np
import pickle as pk
import functions_generic as f_gen
import functions_sampling as f_samp
import functions_numerical as f_nume
import matplotlib
import sys
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot probability density functions.
# Parameters:
# ---DISTR_TYPE: integer value to select a distribution to plot. The options are:
#                polylogarithmic exponential (1); shifted-geometric exponential (2);
#                polylogarithmic and shifted-geometric exponential for joint plot of
#                k-th order approximation (3).
# ---N         : integer value with the number of neurons in the population.
# ---BASE_PARAMS: string with comma separated baseline parameter values for each
#                distribution. For DISTR_TYPE=1 the format is 'f,m,M', where f>0 is the
#                sparsity inducing parameter and the integer m>0 is the polylogarithmic
#                order and M is the number of terms to use for the m>1 approximation.
#                For DISTR_TYPE=2 the format is 'f,tau' where f is as in the DISTR_TYPE=1
#                case and 0<tau<1 refers to the tau shifted-geometric parameter.
# ---OUT_FOLDER: string value with the output folder, where the png image(s) of
#                the plot(s) is(are) required to be stored.
# ---PPARAM_LST: list with plotting parameters. The elements of the list are:
#                COLOR_LIST,LINE_STYLES and DPI, each of which consist of a comma-separated
#                string with exception of the integer DPI. The parameter description are
#                available in the file main.py.
# Returns:
# ---No return value. The image of the plot is stored in the OUT_FOLDER output folder.
def plot_PDF( DISTR_TYPE, N, BASE_PARAMS, OUT_FOLDER, PPARAM_LST ):
   
   # Obtain parameters
   B_PARAMS     = BASE_PARAMS.split(',')
   COLOR_LIST   = PPARAM_LST[0].split(',')
   LINE_STYLES  = PPARAM_LST[1].split(',')
   DPI          = int(PPARAM_LST[2])
   
   size_cols    = len(COLOR_LIST)
   size_lsty    = len(LINE_STYLES)
   
   # Other variables
   eps          = 1e-15
   pdf_r_LIST   = np.zeros((10,N+1),dtype=np.float32)
   pdf_r        = np.zeros((N+1,),dtype=np.float32)
   pdf_r        = pdf_r.copy(order='C')
   r            = np.asarray( np.arange(0,1.0 + float(1. / float(N)), float(1. / float(N)) ) , dtype=np.float32)
   r[0]         = eps
   r[N]         = 1.0 - eps
   r            = r.copy(order='C')
   
   if DISTR_TYPE == 1 or DISTR_TYPE == 3  :
      # -----------------------------------------------------------------------------
      # Polylogarithmic exponential case
      f         = float(B_PARAMS[0])
      m         = int(B_PARAMS[1])
      M         = int(B_PARAMS[2])
      
      # Obtain PDF values for polylogarithmic form varying f
      f_vals    = [1.,5.,10.,15.]
      i         = 0
      for f_vary in f_vals:
         f_gen.polylogarithmic_pdf( pdf_r, r, N+1, f_vary, m, M )
         pdf_r_LIST[i,:] = pdf_r[:]
         i = i + 1
      # Obtain PDF values for polylogarithmic form varying m
      m_vals    = [1,3,5,7]
      
      for m_vary in m_vals:
         f_gen.polylogarithmic_pdf( pdf_r, r, N+1, f, m_vary, M )
         pdf_r_LIST[i,:] = pdf_r[:]
         i = i + 1
      
      # Plot results
      # ------------------------------------------------------
      
      fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
      fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.1, hspace = 0.05)
      
      ind  = 0
      for f_vary in f_vals:
         ax1.plot( r , pdf_r_LIST[ind,:], linestyle=LINE_STYLES[ind % size_lsty], linewidth=2.0, color=COLOR_LIST[ind % size_cols], label='f='+str(f_vary))
         ind = ind + 1
      #ax1.set_title("Polylogarithmic exponential PDF",fontsize=20)
      ax1.set_xlabel('$r$',fontsize=18)
      ax1.set_ylabel('$p(r | f, m)$',fontsize=18)
      ax1.grid()
      ax1.legend(prop={'size': 16},loc='best')
      
      for m_vary in m_vals:
         ax2.plot( r , pdf_r_LIST[ind,:], linestyle=LINE_STYLES[ind % size_lsty], linewidth=2.0, color=COLOR_LIST[ind % size_cols], label='m='+str(m_vary))
         ind = ind + 1
      #ax2.set_title("Polylogarithmic exponential PDF",fontsize=20)
      ax2.set_xlabel('$r$',fontsize=18)
      ax2.set_ylabel("")
      ax2.grid()
      ax2.legend(prop={'size': 16},loc='best')
      fig.suptitle("Polylogarithmic exponential PDF",fontsize=20)
      #for ax in fig.get_axes():
      #    ax.label_outer()
      fig.savefig(OUT_FOLDER+'/PDF_POLYLOGARITHMIC_VARY_f_m.png', bbox_inches = 'tight')
      
      # -----------------------------------------------------------------------------
      
   else:
      # -----------------------------------------------------------------------------
      # Shifted-geometric exponential case
      f         = float(B_PARAMS[0])
      tau       = float(B_PARAMS[1])
      
      # Obtain PDF values for shifted-geometric form varying f
      f_vals    = [1.,5.,10.,15.]
      i         = 0
      for f_vary in f_vals:
         f_gen.shifted_geometric_pdf( pdf_r, r, N+1, f_vary, tau)
         pdf_r_LIST[i,:] = pdf_r[:]
         i = i + 1
      # Obtain PDF values for shifted-geometric form varying tau
      tau_vals  = [0.2,0.4,0.6,0.8]
      for tau_vary in tau_vals:
         f_gen.shifted_geometric_pdf( pdf_r, r, N+1, f, tau_vary)
         pdf_r_LIST[i,:] = pdf_r[:]
         i = i + 1
      
      #Plot results
      # ------------------------------------------------------
      
      fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
      fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.1, hspace = 0.05)
      
      ind  = 0
      for f_vary in f_vals:
         ax1.plot( r , pdf_r_LIST[ind,:], linestyle=LINE_STYLES[ind % size_lsty], linewidth=2.0, color=COLOR_LIST[ind % size_cols], label='f='+str(f_vary))
         ind = ind + 1
      #ax1.set_title("Shifted-geometric exponential PDF",fontsize=20)
      ax1.set_xlabel('$r$',fontsize=18)
      ax1.set_ylabel('$p(r | f, \\tau)$',fontsize=18)
      ax1.grid()
      ax1.legend(prop={'size': 16},loc='best')
      
      for tau_vary in tau_vals:
         ax2.plot( r ,pdf_r_LIST[ind,:], linestyle=LINE_STYLES[ind % size_lsty], linewidth=2.0, color=COLOR_LIST[ind % size_cols], label='$\\tau$='+str(tau_vary))
         ind = ind + 1
      #ax2.set_title("Shifted-geometric exponential PDF",fontsize=20)
      ax2.set_xlabel('$r$',fontsize=18)
      ax2.set_ylabel("")
      ax2.grid()
      ax2.legend(prop={'size': 16},loc='best')
      fig.suptitle("Shifted-geometric exponential PDF",fontsize=20)
      fig.savefig(OUT_FOLDER+'/PDF_SHIFTED_GEOMETRIC_VARY_f_tau.png', bbox_inches = 'tight')
      
      # -----------------------------------------------------------------------------
   
   
   if DISTR_TYPE == 1 :
      # -----------------------------------------------------------------------------
      # Polylogarithmic exponential case
      f         = float(B_PARAMS[0])
      m         = int(B_PARAMS[1])
      M         = int(B_PARAMS[2])
      
      # Polylogarithmic exponential PDF baseline comparison
      # -----------------------------------------------------------------------------
      
      LABS      = ['Baseline','1st. order approx.','2nd. order approx.',\
                   '3rd. order approx.','4th. order approx.']
      f_gen.polylogarithmic_pdf( pdf_r, r, N+1, f, m, M )
      pdf_r_LIST[0,:] = pdf_r[:]
      f_gen.first_order_poly_pdf( pdf_r, r, N+1, f )
      pdf_r_LIST[1,:] = pdf_r[:]
      f_gen.second_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[2,:] = pdf_r[:]
      f_gen.third_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[3,:] = pdf_r[:]
      f_gen.fourth_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[4,:] = pdf_r[:]
      
      #Plot results
      # -----------------------------------------------------------------------------
      fig  = plt.figure(figsize=(12,6),dpi=DPI)
      fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
      ax1  = fig.add_subplot(1,1,1)
      for c in range(0,5):
         ax1.plot( pdf_r_LIST[0,:] , pdf_r_LIST[c,:] , linestyle=LINE_STYLES[c % size_lsty], linewidth=2.0, color=COLOR_LIST[c % size_cols], label=LABS[c])
      ax1.set_title("Polylogarithmic exp. PDF $\;$vs$\;$ k-th order approx. (log-scale)",fontsize=20,pad=20)
      ax1.set_xscale("log")
      ax1.set_yscale("log")
      ax1.set_xlim(xmin=0.03,xmax=10)
      ax1.set_ylim(ymin=0.03,ymax=10)
      #ax1.set_xlabel("exp$[\;f\;Li_m[-r]\;] \;/\;Z$",fontsize=18)
      #ax1.set_ylabel("exp$[-f C_1 r + ... + (-1)^k f C_k r^k\;]\;/\;Z\;$",fontsize=18)
      ax1.set_xlabel("$\;f\;Li_m[-r]\;$ -log($;Z$)",fontsize=18)
      ax1.set_ylabel("-f C_1 r + ... + (-1)^k f C_k r^k\;$-log($\;Z\;$)",fontsize=18)
      ax1.grid()
      ax1.legend(prop={'size': 16},loc='best')
      ax1.set_box_aspect(1)
      fig.savefig(OUT_FOLDER+'/POLYLOGARITHMIC_EXP_KTH_ORDER_APPROX.png', bbox_inches = 'tight')
      
      # -----------------------------------------------------------------------------
      
   elif DISTR_TYPE == 2 :
      # -----------------------------------------------------------------------------
      # Shifted-geometric exponential case
      f         = float(B_PARAMS[0])
      tau       = float(B_PARAMS[1])
      
      # Shifted-geometric exponential PDF baseline comparison
      # -----------------------------------------------------------------------------
      
      LABS      = ['Baseline','1st. order approx.','2nd. order approx.',\
                   '3rd. order approx.','4th. order approx.']
      f_gen.shifted_geometric_pdf( pdf_r, r, N+1, f, tau)
      pdf_r_LIST[0,:] = pdf_r[:]
      f_gen.first_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[1,:] = pdf_r[:]
      f_gen.second_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[2,:] = pdf_r[:]
      f_gen.third_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[3,:] = pdf_r[:]
      f_gen.fourth_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[4,:] = pdf_r[:]
      
      #Plot results
      # -----------------------------------------------------------------------------
      fig  = plt.figure(figsize=(12,6),dpi=DPI)
      fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
      ax1  = fig.add_subplot(1,1,1)
      for c in range(0,5):
         ax1.plot( pdf_r_LIST[0,:] , pdf_r_LIST[c,:] , linestyle=LINE_STYLES[c % size_lsty], linewidth=2.0, color=COLOR_LIST[c % size_cols], label=LABS[c])
      ax1.set_title("Shifted-geometric exp. PDF $\;$vs$\;$ k-th order approx. (log-scale)",fontsize=20,pad=20)
      ax1.set_xscale("log")
      ax1.set_yscale("log")
      ax1.set_xlim(xmin=0.03,xmax=10)
      ax1.set_ylim(ymin=0.03,ymax=10)
      #ax1.set_xlabel("exp$[\;f\;(1/(1+\\tau r) - 1)\;] \;/\;Z$",fontsize=18)
      #ax1.set_ylabel("exp$[-f C_1 r + ... + (-1)^k f C_k r^k\;]\;/\;Z\;$",fontsize=18)
      ax1.set_xlabel("$\;f\;(1/(1+\\tau r) - 1)\;$ -log($\;Z\;$)",fontsize=18)
      ax1.set_ylabel("$-f C_1 r + ... + (-1)^k f C_k r^k\;$ -log($\;Z\;$)",fontsize=18)
      ax1.grid()
      ax1.legend(prop={'size': 16},loc='best')
      ax1.set_box_aspect(1)
      fig.savefig(OUT_FOLDER+'/SHIFTED_GEOMETRIC_EXP_KTH_ORDER_APPROX.png', bbox_inches = 'tight')
      
      # -----------------------------------------------------------------------------
      
   else :
      # -----------------------------------------------------------------------------
      # Polylogarithmic exponential case
      f         = float(B_PARAMS[0])
      m         = int(B_PARAMS[1])
      M         = int(B_PARAMS[2])
      
      # Shifted-geometric exponential case
      tau       = float(B_PARAMS[3])
      
      LABS      = ['Baseline','1st. order approx.','2nd. order approx.',\
                   '3rd. order approx.','4th. order approx.']
      
      #Plot results
      # ------------------------------------------------------
      
      f_gen.polylogarithmic_pdf( pdf_r, r, N+1, f, m, M )
      pdf_r_LIST[0,:] = pdf_r[:]
      f_gen.first_order_poly_pdf( pdf_r, r, N+1, f )
      pdf_r_LIST[1,:] = pdf_r[:]
      f_gen.second_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[2,:] = pdf_r[:]
      f_gen.third_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[3,:] = pdf_r[:]
      f_gen.fourth_order_poly_pdf( pdf_r, r, N+1, f, m )
      pdf_r_LIST[4,:] = pdf_r[:]
      
      fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),dpi=DPI)
      fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.02, hspace = 0.05) #wspace = 0.1, hspace = 0.05)
      
      for c in range(0,5):
         ax1.plot( pdf_r_LIST[0,:] , pdf_r_LIST[c,:] , linestyle=LINE_STYLES[c % size_lsty], linewidth=2.0, color=COLOR_LIST[c % size_cols] )#, label=LABS[c])
      ax1.set_xscale("log")
      ax1.set_yscale("log")
      ax1.set_xlim(xmin=0.03,xmax=10)
      ax1.set_ylim(ymin=0.03,ymax=10)
      ax1.set_xlabel('exp$[\;f\;Li_m[-r]\;] \;/\;Z$',fontsize=18)
      ax1.set_ylabel('exp$[-f C_1 r + ... + (-1)^k f C_k r^k\;]\;/\;Z\;$',fontsize=18)
      ax1.grid()
      ax1.legend().set_visible(False)
      ax1.set_box_aspect(1)
      
      f_gen.shifted_geometric_pdf( pdf_r, r, N+1, f, tau)
      pdf_r_LIST[5,:] = pdf_r[:]
      f_gen.first_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[6,:] = pdf_r[:]
      f_gen.second_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[7,:] = pdf_r[:]
      f_gen.third_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[8,:] = pdf_r[:]
      f_gen.fourth_order_s_geom_pdf( pdf_r, r, N+1, f, tau )
      pdf_r_LIST[9,:] = pdf_r[:]
      
      for c in range(0,5):
         ax2.plot( pdf_r_LIST[5,:] , pdf_r_LIST[5+c,:] , linestyle=LINE_STYLES[c % size_lsty], linewidth=2.0, color=COLOR_LIST[c % size_cols], label=LABS[c])
      #ax2.set_title("Shifted-geometric exponential PDF",fontsize=20)
      ax2.set_xscale("log")
      ax2.set_yscale("log")
      ax2.set_xlim(xmin=0.03,xmax=10)
      ax2.set_ylim(ymin=0.03,ymax=10)
      ax2.set_xlabel('exp$[\;f\;(1/(1+\\tau r) - 1)\;] \;/\;Z$',fontsize=18)
      
      ax2.grid(True)
      for tick in ax2.yaxis.get_minor_ticks():
         tick.tick1line.set_visible(False)
         tick.tick2line.set_visible(False)
      for tick in ax2.yaxis.get_major_ticks():
         tick.tick1line.set_visible(False)
         tick.tick2line.set_visible(False)
         tick.label1.set_visible(False)
         tick.label2.set_visible(False)
      ax2.set_box_aspect(1)
      
      handles, labels = ax2.get_legend_handles_labels()
      fig.legend(handles, labels, prop={'size': 16}, loc='upper right', bbox_to_anchor=(1.05,0.42))
      
      fig.suptitle("True PDF $\;$vs$\;$ k-th order approx. (log-scale)",fontsize=20)
      fig.savefig(OUT_FOLDER+'/POLYLOGARITHMIC_SHIFTED_GEOMETRIC_EXP_KTH_ORDER_APPROX.png', bbox_inches = 'tight')
   
   # ----------------------------------------------------------------------------------------------------------------------------
   
   return None


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to visualize the convergence of either the Q_N(r_N; theta_N) polynomial function
# or the probability mass functions to their continuous versions as N grows.
# Parameters:
# ---VIS_CONV_T: integer value to select the visual convergence test. The options are:
#                polynomial Q_N(r_N; theta_N) (1); probability mass functions
#                P(r_N; theta_N) (2).
# ---DISTR_TYPE: integer value to select a distribution to plot. The options are:
#                polylogarithmic exponential (1); shifted-geometric exponential (2).
# ---N_P       : integer value with the maximum number of points per row for the 2D plots.
#                This value is limited to the range [3,21].
# ---BASE_PARAMS: string with comma separated baseline parameter values for each
#                distribution. For DISTR_TYPE=1 the format is 'f,m,M', where f>0 is the
#                sparsity inducing parameter and the integer m>0 is the polylogarithmic
#                order and M is the number of terms to use for the m>1 approximation.
#                For DISTR_TYPE=2 the format is 'f,tau' where f is as in the DISTR_TYPE=1
#                case and 0<tau<1 refers to the tau shifted-geometric parameter.
# ---OUT_FOLDER: string value with the output folder, where the png image(s) of
#                the plot(s) is(are) required to be stored.
# ---PPARAM_LST: list with plotting parameters. The elements of the list are:
#                COLOR_LIST,LINE_STYLES and DPI, each of which consist of a comma-separated
#                string with exception of the integer DPI. The parameter description are
#                available in the file main.py.
# Returns:
# ---No return value. The image of the plot is stored in the OUT_FOLDER output folder.
def plot_visual_convergence_test( VIS_CONV_T, DISTR_TYPE, N_P, BASE_PARAMS,\
                                  OUT_FOLDER, PPARAM_LST ):
   N_MIN = 3
   N_MAX = 21
   if N_P < N_MIN :
      N_P = N_MIN
      print("plot_visual_convergence_test:: setting N_P (number of points) = 3 (minimum allowed).")
   if N_P > N_MAX :
      N_P = N_MAX
      print("plot_visual_convergence_test:: setting N_P (number of points) = 21 (maximum allowed).")
   # Obtain parameters
   B_PARAMS     = BASE_PARAMS.split(',')
   COLOR_LIST   = PPARAM_LST[0].split(',')
   LINE_STYLES  = PPARAM_LST[1].split(',')
   DPI          = int(PPARAM_LST[2])
   
   size_cols    = len(COLOR_LIST)
   size_lsty    = len(LINE_STYLES)
   
   # Other variables
   eps          = 1e-15
   Abs_DISCR_CONTINUOUS = np.zeros((N_P-N_MIN,N_P),dtype=np.float32)
   Q_r          = np.zeros((N_P,),dtype=np.float32)
   Q_r          = Q_r.copy(order='C')
   THETA        = np.zeros((N_P-1,),dtype=np.float32)
   THETA        = THETA.copy(order='C')
   FACT_k       = np.ones((N_P-1,),dtype=np.float64)
   FACT_k       = FACT_k.copy(order='C')
   
   f_nume.factorial_array( FACT_k, N_P-1 ) # <-- stores factorials up to N_P-1
   
   # --------------------------------------------------------------------------------
   # --------------------------------------------------------------------------------
   if VIS_CONV_T == 1 : # Convergence test for Q_N( r_N ; theta_N ) polynomial
      if DISTR_TYPE == 1 :
         # -----------------------------------------------------------------------------
         # Polylogarithmic exponential case
         f         = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         #M         = int(B_PARAMS[2])
         
         for np_i in range(N_MIN,N_P):
            Q_r    = np.zeros((N_P,),dtype=np.float32)
            Q_r    = Q_r.copy(order='C')
            f_nume.obtain_theta( THETA, FACT_k, np_i-1, DISTR_TYPE, f, m, 0.0 )
            f_nume.Q_polynomial( Q_r, THETA, FACT_k, np_i )
            cell_s = float(N_P / np_i)
            for ci in range(0,np_i-1):
               Abs_DISCR_CONTINUOUS[np_i-N_MIN,int(ci*cell_s):int((ci+1)*cell_s)] = np.abs( Q_r[ci] + f*np.log( 1.0 + float(ci) / float(np_i) ) )
            Abs_DISCR_CONTINUOUS[np_i-N_MIN,int((np_i-1)*cell_s):] = np.abs( Q_r[np_i-1] + f*np.log( 1.0 + float(np_i-1) / float(np_i) ) )
         
         # Plot results
         # ------------------------------------------------------
         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
         #ax.set_axis_off()
         ax.set_title("| $Q_N(r_N ; \\theta_N)$ - (-$f$ log($1+r_N$)) |",fontsize=20,pad=20)
         ax.set_xlabel("$r$",fontsize=18)
         ax.set_ylabel("$N$",fontsize=18)
         im          = ax.imshow(Abs_DISCR_CONTINUOUS, cmap='bwr', \
                                 interpolation='nearest',vmin=Abs_DISCR_CONTINUOUS.min(),\
                                 vmax=Abs_DISCR_CONTINUOUS.max(),\
                                 extent=[0.0,1.0,N_P,N_MIN], aspect = 1.0 / float(N_P-N_MIN) )
         im.set_clim([Abs_DISCR_CONTINUOUS.min(),Abs_DISCR_CONTINUOUS.max()])
         fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                           wspace=0.02, hspace=0.02)
         cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
         cbar = fig.colorbar(im, cax=cb_ax)
         # set the colorbar ticks and tick labels
         cbar.set_ticks([Abs_DISCR_CONTINUOUS.min(), 0.0, Abs_DISCR_CONTINUOUS.max()])
         #cbar.set_ticklabels(['min', '0', 'max'])
         fig.savefig(OUT_FOLDER + '/Q_R_POLYNOMIAL_POLYLOGARITHMIC_F'+str(f)+'_m'+str(m)+'.png')
         
         # -----------------------------------------------------------------------------
         
      else :
         # -----------------------------------------------------------------------------
         # Shifted-geometric exponential case
         f         = float(B_PARAMS[0])
         tau       = float(B_PARAMS[1])
         
         for np_i in range(N_MIN,N_P):
            Q_r    = np.zeros((N_P,),dtype=np.float32)
            Q_r    = Q_r.copy(order='C')
            f_nume.obtain_theta( THETA, FACT_k, np_i-1, DISTR_TYPE, f, 0, tau )
            f_nume.Q_polynomial( Q_r, THETA, FACT_k, np_i )
            cell_s = float(N_P / np_i)
            for ci in range(0,np_i-1):
               Abs_DISCR_CONTINUOUS[np_i-N_MIN,int(ci*cell_s):int((ci+1)*cell_s)] = np.abs( Q_r[ci] -f*( 1. / (1.0 + tau*(float(ci) / float(np_i))) - 1.0 ) )
            Abs_DISCR_CONTINUOUS[np_i-N_MIN,int((np_i-1)*cell_s):] = np.abs( Q_r[np_i-1] -f*( 1. / (1.0 + tau*(float(ci) / float(np_i))) - 1.0 ) )
         
         # Plot results
         # ------------------------------------------------------
         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
         #ax.set_axis_off()
         ax.set_title("| $Q_N(r_N ; \\theta_N)$ - $f(1 / (1 + \\tau r_N) - 1)$ |",fontsize=20,pad=20)
         ax.set_xlabel("$r$",fontsize=18)
         ax.set_ylabel("$N$",fontsize=18)
         im          = ax.imshow(Abs_DISCR_CONTINUOUS, cmap='bwr', \
                                 interpolation='nearest',vmin=Abs_DISCR_CONTINUOUS.min(),\
                                 vmax=Abs_DISCR_CONTINUOUS.max(),\
                                 extent=[0.0,1.0,N_P,N_MIN], aspect = 1.0 / float(N_P-N_MIN) )
         im.set_clim([Abs_DISCR_CONTINUOUS.min(),Abs_DISCR_CONTINUOUS.max()])
         fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                           wspace=0.02, hspace=0.02)
         cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
         cbar = fig.colorbar(im, cax=cb_ax)
         # set the colorbar ticks and tick labels
         cbar.set_ticks([Abs_DISCR_CONTINUOUS.min(), 0.0, Abs_DISCR_CONTINUOUS.max()])
         #cbar.set_ticklabels(['min', '0', 'max'])
         fig.savefig(OUT_FOLDER + '/Q_R_POLYNOMIAL_SHIFTED_GEOMETRIC_F'+str(f)+'_tau'+str(tau)+'.png')
         
         # -----------------------------------------------------------------------------
         
   else:                # Convergence test for P( r_N ; theta_N ) distribution
      pmf_r        = np.zeros((N_P,),dtype=np.float32)
      pmf_r        = pmf_r.copy(order='C')
      pdf_r        = np.zeros((N_P,),dtype=np.float32)
      pdf_r        = pdf_r.copy(order='C')
      Z            = 0.0
      if DISTR_TYPE == 1 :
         # -----------------------------------------------------------------------------
         # Polylogarithmic exponential case
         f         = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         M         = int(B_PARAMS[2])
         
         for np_i in range(N_MIN,N_P):
            Q_r    = np.zeros((N_P,),dtype=np.float32)
            Q_r    = Q_r.copy(order='C')
            f_nume.obtain_theta( THETA, FACT_k, np_i-1, DISTR_TYPE, f, m, 0.0 )
            f_nume.Q_polynomial( Q_r, THETA, FACT_k, np_i )
            
            # Obtain probability mass function values
            for ci in range(0,np_i):
               pmf_r[ ci ] = np.exp( Q_r[ci] )
            Z      = 0.0
            for ci in range(0,np_i):
               Z   = Z + pmf_r[ ci ]
            for ci in range(0,np_i):
               pmf_r[ ci ] = pmf_r[ ci ] / Z
            # Obtain probability density function values
            r            = np.asarray( np.arange(0,1.0 + float(1. / float(np_i)), float(1. / float(np_i)) ) , dtype=np.float32)
            r            = r.copy(order='C')
            f_gen.polylogarithmic_pdf( pdf_r, r, np_i, f, m, M )
            
            cell_s = float(N_P / np_i)
            for ci in range(0,np_i-1):
               Abs_DISCR_CONTINUOUS[np_i-N_MIN,int(ci*cell_s):int((ci+1)*cell_s)] = np.abs( (pmf_r[ ci ]*float(np_i)) - pdf_r[ ci ] )
            Abs_DISCR_CONTINUOUS[np_i-N_MIN,int((np_i-1)*cell_s):] = np.abs( (pmf_r[ np_i-1 ]*float(np_i)) - pdf_r[np_i-1] )
         
         # Plot results
         # ------------------------------------------------------
         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
         #ax.set_axis_off()
         ax.set_title("| $P(r_N ; \\theta_N) N$ - $p(r_N; f, m)$ |",fontsize=20,pad=20)
         ax.set_xlabel("$r$",fontsize=18)
         ax.set_ylabel("$N$",fontsize=18)
         im          = ax.imshow(Abs_DISCR_CONTINUOUS, cmap='bwr', \
                                 interpolation='nearest',vmin=Abs_DISCR_CONTINUOUS.min(),\
                                 vmax=Abs_DISCR_CONTINUOUS.max(),\
                                 extent=[0.0,1.0,N_P,N_MIN], aspect = 1.0 / float(N_P-N_MIN) )
         im.set_clim([Abs_DISCR_CONTINUOUS.min(),Abs_DISCR_CONTINUOUS.max()])
         fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                           wspace=0.02, hspace=0.02)
         cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
         cbar = fig.colorbar(im, cax=cb_ax)
         # set the colorbar ticks and tick labels
         cbar.set_ticks([Abs_DISCR_CONTINUOUS.min(), 0.0, Abs_DISCR_CONTINUOUS.max()])
         #cbar.set_ticklabels(['min', '0', 'max'])
         fig.savefig(OUT_FOLDER + '/PMF_POLYLOGARITHMIC_F'+str(f)+'_m'+str(m)+'.png')
         
         # -----------------------------------------------------------------------------
         
      else :
         # -----------------------------------------------------------------------------
         # Shifted-geometric exponential case
         f         = float(B_PARAMS[0])
         tau       = float(B_PARAMS[1])
         
         for np_i in range(N_MIN,N_P):
            Q_r    = np.zeros((N_P,),dtype=np.float32)
            Q_r    = Q_r.copy(order='C')
            f_nume.obtain_theta( THETA, FACT_k, np_i-1, DISTR_TYPE, f, 0, tau )
            f_nume.Q_polynomial( Q_r, THETA, FACT_k, np_i )
            
            # Obtain probability mass function values
            for ci in range(0,np_i):
               pmf_r[ ci ] = np.exp( Q_r[ci] )
            Z      = 0.0
            for ci in range(0,np_i):
               Z   = Z + pmf_r[ ci ]
            for ci in range(0,np_i):
               pmf_r[ ci ] = pmf_r[ ci ] / Z
            # Obtain probability density function values
            r            = np.asarray( np.arange(0,1.0 + float(1. / float(np_i)), float(1. / float(np_i)) ) , dtype=np.float32)
            r            = r.copy(order='C')
            f_gen.shifted_geometric_pdf( pdf_r, r, np_i, f, tau)
            
            cell_s = float(N_P / np_i)
            for ci in range(0,np_i-1):
               Abs_DISCR_CONTINUOUS[np_i-N_MIN,int(ci*cell_s):int((ci+1)*cell_s)] = np.abs( (pmf_r[ ci ]*float(np_i)) - pdf_r[ci] )
            Abs_DISCR_CONTINUOUS[np_i-N_MIN,int((np_i-1)*cell_s):] = np.abs( (pmf_r[ np_i-1 ]*float(np_i)) - pdf_r[np_i-1] )
         
         # Plot results
         # ------------------------------------------------------
         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
         #ax.set_axis_off()
         ax.set_title("| $P(r_N ; \\theta_N) N$ - $p(r_N; f, \\tau )$ |",fontsize=20,pad=20)
         ax.set_xlabel("$r$",fontsize=18)
         ax.set_ylabel("$N$",fontsize=18)
         im          = ax.imshow(Abs_DISCR_CONTINUOUS, cmap='bwr', \
                                 interpolation='nearest',vmin=Abs_DISCR_CONTINUOUS.min(),\
                                 vmax=Abs_DISCR_CONTINUOUS.max(),\
                                 extent=[0.0,1.0,N_P,N_MIN], aspect = 1.0 / float(N_P-N_MIN) )
         im.set_clim([Abs_DISCR_CONTINUOUS.min(),Abs_DISCR_CONTINUOUS.max()])
         fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                           wspace=0.02, hspace=0.02)
         cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
         cbar = fig.colorbar(im, cax=cb_ax)
         # set the colorbar ticks and tick labels
         cbar.set_ticks([Abs_DISCR_CONTINUOUS.min(), 0.0, Abs_DISCR_CONTINUOUS.max()])
         #cbar.set_ticklabels(['min', '0', 'max'])
         fig.savefig(OUT_FOLDER + '/PMF_SHIFTED_GEOMETRIC_F'+str(f)+'_tau'+str(tau)+'.png')
         
         # -----------------------------------------------------------------------------
         
   # --------------------------------------------------------------------------------
   # --------------------------------------------------------------------------------
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot histograms after simulation from a selected distribution.
# Parameters:
# ---SAMPLING_TYPE: integer value with the type of sampling to consider. The options are
#                continuous population rate distributions (1); N-dimensional binary neurons
#                distributions (2).
# ---DISTR_TYPE: integer value to select a distribution to sample from. The options are:
#                polylogarithmic exponential (1); shifted-geometric exponential (2);
#                polylogarithmic and shifted-geometric exponential for joint plot of
#                the histograms (3).
# ---N_SAMP    : integer value with the number of samples to draw for each distribution.
# ---N_BINS    : integer value with the number of bins for each histogram.
# ---N         : integer value with the number of neurons in the population.
# ---BASE_PARAMS: string with comma separated baseline parameter values for each
#                distribution. For DISTR_TYPE=1 the format is 'f,m,M', where f>0 is the
#                sparsity inducing parameter and the integer m>0 is the polylogarithmic
#                order and M is the number of terms to use for the m>1 approximation.
#                For DISTR_TYPE=2 the format is 'f,tau' where f is as in the DISTR_TYPE=1
#                case and 0<tau<1 refers to the tau shifted-geometric parameter.
# ---OUT_FOLDER: string value with the output folder, where the png image(s) of
#                the plot(s) is(are) required to be stored.
# ---PPARAM_LST: list with plotting parameters. The elements of the list are:
#                COLOR_LIST,LINE_STYLES and DPI, each of which consist of a comma-separated
#                string with exception of the integer DPI. The parameter description are
#                available in the file main.py.
# Returns:
# ---No return value. The image of the plot is stored in the OUT_FOLDER output folder.
def plot_histogram( SAMPLING_TYPE, DISTR_TYPE, N_SAMP,N_BINS,N, BASE_PARAMS,\
                    OUT_FOLDER, PPARAM_LST ):
   
   # Obtain parameters
   B_PARAMS     = BASE_PARAMS.split(',')
   COLOR_LIST   = PPARAM_LST[0].split(',')
   LINE_STYLES  = PPARAM_LST[1].split(',')
   DPI          = int(PPARAM_LST[2])
   
   size_cols    = len(COLOR_LIST)
   size_lsty    = len(LINE_STYLES)
   
   # Other variables
   eps          = 1e-15
   delta_r      = 0.001
   np.random.seed(1010) # set the seed for the pseudo-random numbers
   
   # Precompute vectors with distribution functions evaluations for discrete r points
   N_points     = int( 1.0 / delta_r )
   #print("Number of discrete r points = {}".format(Npoints))
   F_vals       = np.zeros((N_points,),dtype=np.float32)
   F_vals       = F_vals.copy(order='C')
   #print("Number of samples to generate = {}".format(N_SAMP))
   
   if SAMPLING_TYPE == 1 :
      r_samples    = np.zeros((N_SAMP,),dtype=np.float32)
      if DISTR_TYPE == 1 :
         # -----------------------------------------------------------------------------
         # Polylogarithmic exponential case
         f         = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         MTERMS    = int(B_PARAMS[2])
         # Evaluate the polylogarithmic probability distribution function
         f_samp.F_polylogarithmic_eval( F_vals, N_points, delta_r, f, m, MTERMS )
         
      elif DISTR_TYPE == 2:
         # -----------------------------------------------------------------------------
         # Shifted-geometric exponential case
         f         = float(B_PARAMS[0])
         tau       = float(B_PARAMS[1])
         # Evaluate the shifted-geometric probability distribution function
         f_samp.F_shifted_geometric_eval( F_vals, N_points, delta_r, f, tau )
      else :
         r_samples2= np.zeros((N_SAMP,),dtype=np.float32)
         # Polylogarithmic exponential case
         f         = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         MTERMS    = int(B_PARAMS[2])
         tau       = float(B_PARAMS[3])
         # Evaluate the polylogarithmic probability distribution function
         f_samp.F_polylogarithmic_eval( F_vals, N_points, delta_r, f, m, MTERMS )
         F_vals2   = np.zeros((N_points,),dtype=np.float32)
         F_vals2   = F_vals2.copy(order='C')
         # Evaluate the shifted-geometric probability distribution function
         f_samp.F_shifted_geometric_eval( F_vals2, N_points, delta_r, f, tau )
         # Apply the Inverse Transform Method for shifted-geometric case
         f_samp.InverseTransform( r_samples2, F_vals2, N_SAMP, N_points, delta_r )
         
      # Apply the Inverse Transform Method to draw samples from the continuous
      # probability density functions
      # --------------------------------------------------------------------------------
      f_samp.InverseTransform( r_samples, F_vals, N_SAMP, N_points, delta_r )
      
   else:
      r_samples    = np.zeros((N_SAMP,),dtype=np.int32)
      r_samples    = r_samples.copy(order='C')
      if DISTR_TYPE == 1 :
         # -----------------------------------------------------------------------------
         # Polylogarithmic exponential case
         f         = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         M_TERMS   = int(B_PARAMS[2])
         # Draw samples from discrete binary version of polylogarithmic distribution
         # with Gibbs sampling (only efficient for m=1)
         f_samp.GibbsSampling_polylogarithmic_hist( r_samples, 10, N, N_SAMP,M_TERMS, f, m )
         
      else:
         # -----------------------------------------------------------------------------
         # Shifted-geometric exponential case
         f         = float(B_PARAMS[0])
         tau       = float(B_PARAMS[1])
         # Draw samples from the discrete binary version of shifted-geometric
         # distribution with Gibbs sampling
         f_samp.GibbsSampling_shifted_geometric_hist( r_samples, 10, N, N_SAMP, f, tau )
      
   # Show plot with the histogram of the samples
   # --------------------------------------------------------------------------------
   if DISTR_TYPE == 3 and SAMPLING_TYPE == 1 :
      fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,12),dpi=DPI)
      fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.02, hspace = 0.15)
      
      ax1.hist( r_samples, density=False, bins=N_BINS, histtype='bar', color=COLOR_LIST[0], alpha=0.7, edgecolor='black', linewidth=1.2 )
      ax1.set_title('Polylogarithmic exp. distribution histogram, $m='+str(m)+'$,$f='+str(f)+'$',fontsize=20,pad=20)
      str_suffix = ''
      if SAMPLING_TYPE == 1 :
         ax1.set_xlabel('',fontsize=18)
         common_y_lab = 'Count of $r$'
         ax1.set_ylabel('',fontsize=18)
      else :
         str_suffix = 'BINARY'
         ax1.set_xlabel('',fontsize=18)
         common_y_lab = 'Count per bin'
         ax1.set_ylabel('',fontsize=18)
      ax1.grid()
      ax1.legend().remove()
      for tick in ax1.xaxis.get_major_ticks():
         tick.tick1line.set_visible(False)
         tick.tick2line.set_visible(False)
         tick.label1.set_visible(False)
         tick.label2.set_visible(False)
      
      
      ax2.hist( r_samples2, density=False, bins=N_BINS, histtype='bar', color=COLOR_LIST[0], alpha=0.7, edgecolor='black', linewidth=1.2 )
      ax2.set_title('Shifted-geometric exp. distribution histogram, $\\tau='+str(tau)+'$,$f='+str(f)+'$',fontsize=20,pad=20)
      str_suffix = ''
      if SAMPLING_TYPE == 1 :
         ax2.set_xlabel('$r$',fontsize=18)
         ax2.set_ylabel('',fontsize=18)
      else :
         str_suffix = 'BINARY'
         ax2.set_xlabel('$n$ (number of active neurons)',fontsize=18)
         ax2.set_ylabel('',fontsize=18)
      ax2.grid()
      ax2.legend().remove()
      fig.text(0.02, 0.5, common_y_lab, ha='center', va='center', rotation='vertical', fontsize=18)
      fig.savefig(OUT_FOLDER+'/HIST_SAMPLES_POLY_SHIFTGEOM_m'+str(m)+'_f'+str(f)+'_tau'+str(tau)+str_suffix+'.png')
      
   else:
      fig    = plt.figure(figsize=(12,6),dpi=DPI)
      fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
      ax1    = fig.add_subplot(1,1,1)
      
      ax1.hist( r_samples, density=False, bins=N_BINS, histtype='bar', color=COLOR_LIST[0], alpha=0.7, edgecolor='black', linewidth=1.2 )
      
      if DISTR_TYPE == 1 :
         ax1.set_title('Polylogarithmic exp. distribution histogram, $m='+str(m)+'$,$f='+str(f)+'$',fontsize=20,pad=20)
      else :
         ax1.set_title('Shifted-geometric exp. distribution histogram, $\\tau='+str(tau)+'$,$f='+str(f)+'$',fontsize=20,pad=20)
      
      str_suffix = ''
      if SAMPLING_TYPE == 1 :
         ax1.set_xlabel('$r$',fontsize=18)
         ax1.set_ylabel('Count of $r$',fontsize=18)
      else :
         str_suffix = 'BINARY'
         ax1.set_xlabel('$n$ (number of active neurons)',fontsize=18)
         ax1.set_ylabel('Count per bin',fontsize=18)
      ax1.grid()
      ax1.legend().remove()
      
      if DISTR_TYPE == 1 :
         fig.savefig(OUT_FOLDER+'/HIST_SAMPLES_POLYLOGARITHM_m'+str(m)+'_f'+str(f)+str_suffix+'.png')
      else :
         fig.savefig(OUT_FOLDER+'/HIST_SAMPLES_SHIFTED_GEOMETRIC_tau'+str(tau)+'_f'+str(f)+str_suffix+'.png')
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to fit the model(s) parameters from a selected distribution to a given dataset under
# the maximum likelihood principle. This function also shows a plot of the fitted model(s) on top
# of the data histogram.
# Parameters:
# ---DISTR_TYPE: integer value to select a distribution to sample from. The options are:
#                polylogarithmic exponential (1); shifted-geometric exponential (2);
#                independent exponential (3); second-order exponential (4); all models comparison with
#                given parameters after learning (5).
# ---N_SAMP    : integer value with the number of samples to draw for each distribution.
# ---N_BINS    : integer value with the number of bins for each histogram.
# ---BASE_PARAMS: string with comma separated baseline parameter values for each
#                distribution. For DISTR_TYPE=1 the format is 'f_init,m,M_TERMS,eta,MAX_ITE,T',
#                where f_init>0 is the initial value of the sparsity inducing parameter and the integers
#                m is the polylogarithmic order parameter, M_TERMS is the number of terms to use for the
#                case of the m>1 approximation, eta is the learning rate for the f parameter, MAX_ITE is
#                the maximum number of global optimization iterations and T is the number of local
#                optimization iterations.
#                For DISTR_TYPE=2 the format is 'f_init,tau,eta,MAX_ITE,T', where f_init>0 is
#                the initial value of the sparsity inducing parameter, tau is the shifted-geometric 
#                parameter, eta is the learning rate for the f parameter, MAX_ITE is the maximum number
#                of global optimization iterations and T is the number of local optimization iterations.
#                For DISTR_TYPE=3 the format is 'f_init,eta,MAX_ITE,T', where the parameter names are the
#                same as in the previous two cases.
#                For DISTR_TYPE=4 the format is 'f_init,eta1,eta2,MAX_ITE,T', where the parameter names
#                are the same as in the previous two cases but f_init is now used to initialize both the
#                f_1 and the f_2 parameters of the second-order model.
#                For DISTR_TYPE=5 the format is 'f_ind,f1_so,f2_so,f_poly,m,M_TERMS,f_sg,tau', where the
#                parameter names are the (learned) sparsity-inducing parameters for the independent model,
#                the second-order model, the polylogarithmic model (followed by its m integer order parameter
#                and the number of terms to use for the m>1 approximation), and the shifted-geometric model
#                (followed by its tau parameter).
# ---IN_FILES:   list of pickle file names for the binary spiking data.
# ---IN_FOLDER:  string value with the input folder, where the spiking data pickle files
#                are located.
# ---OUT_FOLDER: string value with the output folder, where the png image(s) of
#                the plot(s) is(are) required to be stored.
# ---PPARAM_LST: list with plotting parameters. The elements of the list are:
#                COLOR_LIST,LINE_STYLES and DPI, each of which consist of a comma-separated
#                string with exception of the integer DPI. The parameter description are
#                available in the file main.py.
# Returns:
# ---No return value. The image of the plot is stored in the OUT_FOLDER output folder.
def plot_model_fit( DISTR_TYPE, N_SAMP,N_BINS, BASE_PARAMS, IN_FOLDER, IN_FILES, OUT_FOLDER, PPARAM_LST ):
   
   # Obtain parameters
   B_PARAMS     = BASE_PARAMS.split(',')
   COLOR_LIST   = PPARAM_LST[0].split(',')
   LINE_STYLES  = PPARAM_LST[1].split(',')
   DPI          = int(PPARAM_LST[2])
   
   size_cols    = len(COLOR_LIST)
   size_lsty    = len(LINE_STYLES)
   eps          = 1e-15
   
   # Read binary spiking data
   # ---------------------------------------------------------------------------------------
   for file_n in IN_FILES:
      full_n = ''
      if sys.platform.startswith('win'):
         full_n = IN_FOLDER + '\\' + file_n
      else:
         full_n = IN_FOLDER + '/' + file_n
      X      = np.asarray( pk.load( open( full_n, 'rb' ) ) , dtype=np.int32 )
      Ns     = X.shape[0]
      N      = X.shape[1]
      # Sub-sample for faster fit and comparable scale
      Ns_t   = 0
      if N_SAMP > Ns :
         N_SAMP= Ns
         print("Number of samples set at the maximum available of {}".format(Ns))
      ind    = np.arange(0,Ns)
      np.random.seed(1001)
      ind_s  = np.random.choice( ind, size=N_SAMP, replace=False)
      Ns_test= int( 0.2 * N_SAMP )
      Ns     = N_SAMP - Ns_test
      print("Number of samples: {} (training), {} (testing)".format(Ns,Ns_test))
      X_test = X[ ind_s[Ns:(Ns+Ns_test)], :]
      X      = X[ ind_s[0:Ns], :]
      R      = np.zeros((Ns,),dtype=np.float32)
      R_test = np.zeros((Ns_test,),dtype=np.float32)
      
      max_per_sample = 0
      for i in range(0,Ns):
         n_act       = np.sum( X[i,:] )
         R[ i ]      = float( n_act ) / float( N )
         if n_act > max_per_sample :
            max_per_sample = np.sum( X[i,:] )
      for i in range(0,Ns_test):
         n_act       = np.sum( X_test[i,:] )
         R_test[ i ] = float( n_act ) / float( N )
      print("max_per_sample = {}".format(max_per_sample))
      
      X      = X.reshape(-1,)
      X      = X.copy(order='C')
      R      = R.copy(order='C')
      R_test = R_test.copy(order='C')
      np.random.seed(1001)
      np.random.shuffle(R)
      np.random.shuffle(R_test)
      r_do   = np.asarray( np.arange(0,1.0 + float(1. / float(N)), float(1. / float(N)) ) , dtype=np.float32)
      r_do[0]= eps
      r_do[N]= 1.0 - eps
      r_do   = r_do.copy(order='C')
      pdf_r  = np.zeros((N+1,),dtype=np.float32)
      pdf_r  = pdf_r.copy(order='C')
      print("Processing file {} with {} samples and {} neurons ...".format(file_n, Ns, N))
      
      if DISTR_TYPE == 1 :
         # -----------------------------------------------------------------------------
         # Polylogarithmic exponential case
         f_init    = float(B_PARAMS[0])
         m         = int(B_PARAMS[1])
         M_TERMS   = int(B_PARAMS[2])
         eta       = float(B_PARAMS[3])
         MAX_ITE   = int(B_PARAMS[4])
         T         = int(B_PARAMS[5])
         
         # Fit the parameters of the polylogarithmic exp. distr. to the data
         f         = f_nume.model_fit_polylogarithmic_r(  R, r_do, N,Ns,M_TERMS, eta, f_init,m, MAX_ITE,T )
         # Compute analytic solution with found parameters
         f_gen.polylogarithmic_pdf( pdf_r, r_do, N+1, f, m, M_TERMS )
         
         # Compute average ML on test set for each model
         AVG_ML    = f_nume.avg_ml_poly_r( R_test, Ns_test, f, m, M_TERMS )
         
      elif DISTR_TYPE == 2:
         # -----------------------------------------------------------------------------
         # Shifted-geometric exponential case
         f_init    = float(B_PARAMS[0])
         tau       = float(B_PARAMS[1])
         eta       = float(B_PARAMS[2])
         MAX_ITE   = int(B_PARAMS[3])
         T         = int(B_PARAMS[4])
         
         # Fit the parameters of the shifted-geometric exp. distr. to the data
         f         = f_nume.model_fit_shifted_geom_r(  R, r_do, N,Ns, eta, f_init,tau, MAX_ITE,T )
         # Compute analytic solution with found parameters
         f_gen.shifted_geometric_pdf( pdf_r, r_do, N+1, f, tau)
         
         # Compute average ML on test set for each model
         AVG_ML    = f_nume.avg_ml_sg_r( R_test, Ns_test, f, tau )
         
      elif DISTR_TYPE == 3:
         # -----------------------------------------------------------------------------
         # Independent exponential case
         f_init    = float(B_PARAMS[0])
         eta       = float(B_PARAMS[1])
         MAX_ITE   = int(B_PARAMS[2])
         T         = int(B_PARAMS[3])
         # Fit the parameter of the 1st order interactions exp. distr.
         # (truncated exp. distr.) to the data
         # ---------------------------------------------------------------------
         f         = f_nume.model_fit_first_ord_r(  R, r_do, N,Ns, eta, f_init, MAX_ITE,T )
         # Compute analytic solution with found parameters
         f_gen.first_ord_pdf( pdf_r, r_do, N+1, f )
         
         # Compute average ML on test set for each model
         AVG_ML    = f_nume.avg_ml_fo_r( R_test, Ns_test, f )
         
      elif DISTR_TYPE == 4:
         # -----------------------------------------------------------------------------
         # Second-order exponential case
         f_init    = float(B_PARAMS[0])
         eta1      = float(B_PARAMS[1])
         eta2      = float(B_PARAMS[2])
         MAX_ITE   = int(B_PARAMS[3])
         T         = int(B_PARAMS[4])
         # Fit the parameter of the 2nd order interactions exp. distr. to the
         # data
         # ---------------------------------------------------------------------
         f1, f2    = f_nume.model_fit_second_ord_r(  R, r_do, N,Ns, eta1, eta2, f_init, MAX_ITE,T )
         
         # Compute analytic solution with found parameters
         f_gen.second_ord_pdf( pdf_r, r_do, N+1, f1,f2 )
         
         # Compute average ML on test set for each model
         AVG_ML    = f_nume.avg_ml_so_r( R_test, Ns_test, f1, f2 )
         
      else :
         # -----------------------------------------------------------------------------
         # All distributions case for comparison (no fitting procedure)
         f_ind     = float(B_PARAMS[0])
         f1_so     = float(B_PARAMS[1])
         f2_so     = float(B_PARAMS[2])
         f_poly    = float(B_PARAMS[3])
         m         = int(B_PARAMS[4])
         M_TERMS   = int(B_PARAMS[5])
         f_sg      = float(B_PARAMS[6])
         tau       = float(B_PARAMS[7])
         
         # Compute analytic polylogarithmic exp. distr. solution with found parameters
         f_gen.polylogarithmic_pdf( pdf_r, r_do, N+1, f_poly, m, M_TERMS )
         
         # Compute analytic shifted-geometric exp. distr. solution with found parameters
         pdf_r2    = np.zeros((N+1,),dtype=np.float32)
         pdf_r2    = pdf_r2.copy(order='C')
         f_gen.shifted_geometric_pdf( pdf_r2, r_do, N+1, f_sg, tau)
         
         # Compute analytic 1st order interactions exp. distr. solution with found parameters
         pdf_r3    = np.zeros((N+1,),dtype=np.float32)
         pdf_r3    = pdf_r3.copy(order='C')
         f_gen.first_ord_pdf( pdf_r3, r_do, N+1, f_ind )
         
         # Compute analytic 2nd order interactions exp. distr. solution with found parameters
         pdf_r4    = np.zeros((N+1,),dtype=np.float32)
         pdf_r4    = pdf_r4.copy(order='C')
         f_gen.second_ord_pdf( pdf_r4, r_do, N+1, f1_so,f2_so )
         
         # Compute average ML on test set for each model
         AVG_ML_poly   = f_nume.avg_ml_poly_r( R_test, Ns_test, f_poly, m, M_TERMS )
         AVG_ML_sg     = f_nume.avg_ml_sg_r( R_test, Ns_test, f_sg, tau )
         AVG_ML_fo     = f_nume.avg_ml_fo_r( R_test, Ns_test, f_ind )
         AVG_ML_so     = f_nume.avg_ml_so_r( R_test, Ns_test, f1_so, f2_so )
         
      # Show plot with the histogram of the spiking data
      # --------------------------------------------------------------------------------
      if DISTR_TYPE == 5 :
         fig, (ax1, ax2) = plt.subplots( nrows=1, ncols=2, figsize=(14, 6), dpi=DPI )
         fig.subplots_adjust(wspace = 0.3, hspace = 0.2)
         
         # Create histograms
         r_lim     = (max_per_sample+1.)/float(N)
         hist_data = np.zeros((N_BINS,),dtype=np.float32)
         for i in range(0,Ns):
            ind    = int( float(R[i] + eps)*(N_BINS-1))
            hist_data[ ind ] = hist_data[ ind ] + 1
         
         hist_data = hist_data / np.sum( hist_data )
         i_min     = r_do.shape[0]
         if pdf_r.shape[0] < i_min :
            i_min= pdf_r.shape[0]
         # Plot histograms in log-scale
         range_r   = np.asarray( np.arange(0,1.0 + float(1. / float(N_BINS-1)), float(1. / float(N_BINS-1)) ) , dtype=np.float32)
         i_min2    = range_r.shape[0]
         if hist_data.shape[0] < i_min2 :
            i_min2 = hist_data.shape[0]
         
         pdf_r[:i_min]  = pdf_r[:i_min]  / np.sum( pdf_r )
         pdf_r2[:i_min] = pdf_r2[:i_min]  / np.sum( pdf_r2 )
         pdf_r3[:i_min] = pdf_r3[:i_min]  / np.sum( pdf_r3 )
         pdf_r4[:i_min] = pdf_r4[:i_min]  / np.sum( pdf_r4 )
         
         i_min3    = 0
         for i in range(0,r_do.shape[0]):
            if r_do[ i_min3 ] <= r_lim :
               i_min3 = i
            else :
               break
         
         p_min     = min(pdf_r[:i_min3].min(), min(pdf_r2[:i_min3].min(),min(pdf_r3[:i_min3].min(),pdf_r4[:i_min3].min())))
         p_max     = max(pdf_r[:i_min3].max(), max(pdf_r2[:i_min3].max(),max(pdf_r3[:i_min3].max(),pdf_r4[:i_min3].max())))
         
         alpha_lev = 0.7
         ax1.loglog( range_r[:i_min3], hist_data[:i_min3], label='Data points', marker='o', color=COLOR_LIST[0], alpha=alpha_lev, linestyle='None')
         ax1.loglog( r_do[:i_min3], pdf_r[:i_min3], label='Polylogarithmic', marker='.', color=COLOR_LIST[1%size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[1%size_lsty] )
         ax1.loglog( r_do[:i_min3], pdf_r2[:i_min3], label='Shifted-geometric', marker='x', color=COLOR_LIST[2%size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[2%size_lsty] )
         ax1.loglog( r_do[:i_min3], pdf_r3[:i_min3], label='First-order', marker='x', color=COLOR_LIST[3%size_cols], alpha=alpha_lev, linewidth=1.5, linestyle=LINE_STYLES[3%size_lsty] )
         ax1.loglog( r_do[:i_min3], pdf_r4[:i_min3], label='Second-order', marker='+', color=COLOR_LIST[4%size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[4%size_lsty] )
         ax1.set_xscale('linear')
         ax1.set_xlim(0, r_lim)
         ax1.legend()
         alpha_lev = 0.8
         # Plot histograms in linear scale
         ax2.bar( range_r[:i_min2], hist_data[:i_min2], width=np.diff(range_r)[0], alpha=alpha_lev, color=COLOR_LIST[0], edgecolor='black', label='Data histogram' )
         ax2.plot( r_do[:i_min], pdf_r[:i_min], color=COLOR_LIST[1 % size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[1 % size_lsty], label='Polylogarithmic' )
         ax2.plot( r_do[:i_min], pdf_r2[:i_min], color=COLOR_LIST[2 % size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[2 % size_lsty], label='Shifted-geometric' )
         ax2.plot( r_do[:i_min], pdf_r3[:i_min], color=COLOR_LIST[3 % size_cols], alpha=alpha_lev, linewidth=1.5, linestyle=LINE_STYLES[3 % size_lsty], label='First-order' )
         ax2.plot( r_do[:i_min], pdf_r4[:i_min], color=COLOR_LIST[4 % size_cols], alpha=alpha_lev, linewidth=3.0, linestyle=LINE_STYLES[4 % size_lsty], label='Second-order' )
         ax2.set_xlim(0, r_lim)
         
         ax1.set_title('Models fit (log-scale)',fontsize=20,pad=20)
         ax2.set_title('Models fit',fontsize=20,pad=20)
         
         ax1.set_xlabel('$r$ (population rate)',fontsize=18)
         ax1.set_ylabel('Count per bin (Log-scale)',fontsize=18)
         ax1.grid()
         
         ax2.set_xlabel('$r$ (population rate)',fontsize=18)
         ax2.set_ylabel('Normalized model probability',fontsize=18)
         ax2.legend()
         ax2.grid()
         
         # Write average marginal likelihood on test set results on output file
         output_txt_file = OUT_FOLDER+'/MODELS_ALL_POLY_m'+str(m)+'_f'+str(round(f_poly,2))+'_SG_tau'+str(round(tau,2))+'_f'+str(round(f_sg,2))+'_IND_f'+str(round(f_ind,2))+'_SO_f1'+str(round(f1_so,2))+'_f2'+str(round(f2_so,2))+'_'+file_n+'.txt'
         txt_content     = "AVERAGE MARGINAL LIKELIHOOD FOR TESTING SET:\n"
         txt_content     = txt_content + "        "+str(AVG_ML_poly)+" (Polylogarithmic)\n"
         txt_content     = txt_content + "        "+str(AVG_ML_sg)+" (Shifted-geometric)\n"
         txt_content     = txt_content + "        "+str(AVG_ML_fo)+" (First-order)\n"
         txt_content     = txt_content + "        "+str(AVG_ML_so)+" (Second-order)"
         # Write content to the output file
         with open(output_txt_file, 'w') as file:
            file.write(txt_content)
         
         fig.savefig(OUT_FOLDER+'/MODELS_ALL_POLY_m'+str(m)+'_f'+str(round(f_poly,2))+'_SG_tau'+str(round(tau,2))+'_f'+str(round(f_sg,2))+'_IND_f'+str(round(f_ind,2))+'_SO_f1'+str(round(f1_so,2))+'_f2'+str(round(f2_so,2))+'_'+file_n+'.png')
         
      else:
         fig, (ax1, ax2) = plt.subplots( nrows=1, ncols=2, figsize=(20, 6), dpi=DPI )
         fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
         
         # Create histograms
         r_lim     = (max_per_sample+1.)/float(N)
         hist_data = np.zeros((N_BINS,),dtype=np.float32)
         for i in range(0,Ns):
            ind    = int( float(R[i] + eps)*(N_BINS-1))
            hist_data[ ind ] = hist_data[ ind ] + 1
         
         hist_data = hist_data / np.sum( hist_data )
         i_min     = r_do.shape[0]
         if pdf_r.shape[0] < i_min :
            i_min= pdf_r.shape[0]
         # Plot histograms in log-scale
         range_r   = np.asarray( np.arange(0,1.0 + float(1. / float(N_BINS-1)), float(1. / float(N_BINS-1)) ) , dtype=np.float32)
         i_min2    = range_r.shape[0]
         if hist_data.shape[0] < i_min2 :
            i_min2 = hist_data.shape[0]
         ax1.loglog( range_r[:i_min2], hist_data[:i_min2], label='Data points', marker='o', color=COLOR_LIST[0], alpha=0.7, linestyle='None' )
         ax1.loglog( r_do[:i_min], pdf_r[:i_min] / np.sum(pdf_r) , label='PDF model values', marker='o', color=COLOR_LIST[1], alpha=0.7, linewidth=3.0, linestyle=LINE_STYLES[1] )
         ax1.set_xlim(0, r_lim)
         ax1.set_xscale('linear')
         ax1.legend()
         
         # Plot histograms in linear scale
         ax2.bar( range_r[:i_min2], hist_data[:i_min2], width=np.diff(range_r)[0], alpha=0.7, color=COLOR_LIST[0], edgecolor='black', label='Data histogram' )
         ax2.plot( r_do[:i_min], pdf_r[:i_min] / np.sum(pdf_r) , color=COLOR_LIST[1], alpha=0.7, linewidth=3.0, linestyle=LINE_STYLES[1], label='Model probability' )
         ax2.set_xlim(0, r_lim)
         
         if DISTR_TYPE == 1 :
            ax1.set_title('Polylogarithmic exp. distr. model fit (log-scale), $m='+str(m)+'$,$f='+str(round(f,2))+'$',fontsize=20,pad=20)
            ax2.set_title('Polylogarithmic exp. distr. model fit, $m='+str(m)+'$,$f='+str(round(f,2))+'$',fontsize=20,pad=20)
         elif DISTR_TYPE == 2 :
            ax1.set_title('Shifted-geometric exp. distr. model fit (log-scale), $\\tau='+str(round(tau,2))+'$,$f='+str(round(f,2))+'$',fontsize=20,pad=20)
            ax2.set_title('Shifted-geometric exp. distr. model fit, $\\tau='+str(round(tau,2))+'$,$f='+str(round(f,2))+'$',fontsize=20,pad=20)
         elif DISTR_TYPE == 3 :
            ax1.set_title('Independent exp. distr. model fit (log-scale), $f='+str(round(f,2))+'$',fontsize=20,pad=20)
            ax2.set_title('Independent exp. distr. model fit, $f='+str(round(f,2))+'$',fontsize=20,pad=20)
         else :
            ax1.set_title('Second-order exp. distr. model fit (log-scale), $f1='+str(round(f1,2))+'$,$f2='+str(round(f2,2))+'$',fontsize=20,pad=20)
            ax2.set_title('Second-order exp. distr. model fit, $f1='+str(round(f1,2))+'$,$f2='+str(round(f2,2))+'$',fontsize=20,pad=20)
         
         str_suffix = '_POP_RATE'
         ax1.set_xlabel('$r$ (population rate)',fontsize=18)
         ax1.set_ylabel('Count per bin (Log-scale)',fontsize=18)
         ax1.grid()
         
         ax2.set_xlabel('$r$ (population rate)',fontsize=18)
         ax2.set_ylabel('Normalized model probability',fontsize=18)
         ax2.grid()
         
         if DISTR_TYPE == 1 :
            output_txt_file = OUT_FOLDER+'/MODEL_FIT_POLYLOGARITHM_m'+str(m)+'_f'+str(round(f,2))+str_suffix+'_'+file_n+'.txt'
            fig.savefig(OUT_FOLDER+'/MODEL_FIT_POLYLOGARITHM_m'+str(m)+'_f'+str(round(f,2))+str_suffix+'_'+file_n+'.png')
         elif DISTR_TYPE == 2 :
            output_txt_file = OUT_FOLDER+'/MODEL_FIT_SHIFTED_GEOMETRIC_tau'+str(round(tau,2))+'_f'+str(round(f,2))+str_suffix+'_'+file_n+'.txt'
            fig.savefig(OUT_FOLDER+'/MODEL_FIT_SHIFTED_GEOMETRIC_tau'+str(round(tau,2))+'_f'+str(round(f,2))+str_suffix+'_'+file_n+'.png')
         elif DISTR_TYPE == 3 :
            output_txt_file = OUT_FOLDER+'/MODEL_FIT_INDEPENDENT_f'+str(round(f,2))+str_suffix+'_'+file_n+'.txt'
            fig.savefig(OUT_FOLDER+'/MODEL_FIT_INDEPENDENT_f'+str(round(f,2))+str_suffix+'_'+file_n+'.png')
         else :
            output_txt_file = OUT_FOLDER+'/MODEL_FIT_SECOND_ORDER_f1'+str(round(f1,2))+'_f2'+str(round(f2,2))+str_suffix+'_'+file_n+'.txt'
            fig.savefig(OUT_FOLDER+'/MODEL_FIT_SECOND_ORDER_f1'+str(round(f1,2))+'_f2'+str(round(f2,2))+str_suffix+'_'+file_n+'.png')
         
         txt_content     = "AVERAGE MARGINAL LIKELIHOOD FOR TESTING SET:\n"
         txt_content     = txt_content + "        "+str(AVG_ML)
         # Write average marginal likelihood on test set results on output file
         # Write content to the output file
         with open(output_txt_file, 'w') as file:
            file.write(txt_content)
   
   return None
