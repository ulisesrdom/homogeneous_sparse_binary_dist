import numpy as np
import math
#import scipy.special as sc
import functions_numerical as f_nume
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

def Q_tilde(s_not_k, f, distr_type, N, m, tau):
   q_tilde = 0.0
   S1      = 0.0
   S2      = 0.0
   if distr_type == 1 : # polylogarithmic
      for j in range(1,N+1):
         cj = 1. / math.pow(j, m)
         S1 = S1 + (math.pow(-1,j+1)*cj*math.pow( (1. + s_not_k) / float(N) ,j))
         S2 = S2 + (math.pow(-1,j+1)*cj*math.pow( (s_not_k) / float(N) ,j))
   else:                # shifted-geometric
      for j in range(1,N+1):
         cj = math.pow(tau, j)
         S1 = S1 + (math.pow(-1,j+1)*cj*math.pow( (1. + s_not_k) / float(N) ,j))
         S2 = S2 + (math.pow(-1,j+1)*cj*math.pow( (s_not_k) / float(N) ,j))
   q_tilde  = -f * (S1 - S2)
   return q_tilde


# Function to compute the probability of one more neuron becoming
# active given that there are n active neurons.
def prob_k_act_given_n_act( theta, n, N ):
   P        = 0.0
   S_XK_1   = 0.0
   for k in range(1,n+1):
      bi_coeff = 1.0
      for j in range(0,k):
         bi_coeff = bi_coeff * float(n+1-j) / float(k-j)
      S_XK_1   = S_XK_1 + (bi_coeff * theta[k-1])
   S_XK_1   = float( n + 1 ) * math.exp( S_XK_1 )
   S_XK_0   = 0.0
   for k in range(1,n):
      bi_coeff = 1.0
      for j in range(0,k):
         bi_coeff = bi_coeff * float(n-j) / float(k-j)
      S_XK_0   = S_XK_0 + (bi_coeff * theta[k-1])
   S_XK_0   = float( N - n ) * math.exp( S_XK_0 )
   
   P        = S_XK_1 / ( S_XK_1 + S_XK_0 )
   return P


N             = 20
N_f           = 5
f_vals        = [10.,20.,30.,40.]
m             = 5
tau           = 0.8
DISTRIB_TYPE  = 1

FACT_k        = np.ones((N,),dtype=np.float64)
FACT_k        = FACT_k.copy(order='C')
f_nume.factorial_array( FACT_k, N ) # <-- stores factorials up to N

PROB_VALS     = np.zeros((N_f,N,),dtype=np.float32)
ind           = 0
for f in f_vals:
 THETA        = np.zeros((N,),dtype=np.float32)
 THETA        = THETA.copy(order='C')
 f_nume.obtain_theta( THETA, FACT_k, N, DISTRIB_TYPE, f, m, tau )
 for n in range(0,N):
    PROB_VALS[ind,n] = prob_k_act_given_n_act( THETA, n, N )
 print("Probs (f={}) = {}".format(f,PROB_VALS[ind,:]))
 ind          = ind + 1

DPI       = 500
OUT_FOLDER= '.'

#Plot results
# -----------------------------------------------------------------------------
fig, (ax1) = plt.subplots(1,1,figsize=(10,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.1, hspace = 0.05)
n_range    = np.arange(0,N)
colors     = ['tab:blue','tab:orange','tab:green','tab:red','tab:brown']

ind        = 0
for f in f_vals:
   ax1.plot( n_range, PROB_VALS[ind,:] , linestyle='dashed', linewidth=2.0, color=colors[ind%len(colors)], label='$f='+str(f)+'$')
   ind     = ind + 1

ax1.set_xticks(np.arange(0,N,2))
if DISTRIB_TYPE == 1 :
   ax1.set_title("Polylogarithmic exponential distr.",fontsize=20,pad=20)
else:
   ax1.set_title("Shifted-geometric exponential distr.",fontsize=20,pad=20)
ax1.set_xlabel("n",fontsize=18)
ax1.set_ylabel("P( $x_k$ = 1 | \"n active\" )",fontsize=18)
ax1.grid()
ax1.legend(prop={'size': 16},loc='best')
#ax1.set_box_aspect(1)
fig.suptitle("",fontsize=20)
fig.savefig(OUT_FOLDER+'/PROB_N_p_1_ACTIVE_N'+str(N)+'_DISTR'+str(DISTRIB_TYPE)+'.png', bbox_inches = 'tight')



# ----------------------------------CONDITIONAL PROBABILITY -------------------------------------------------
N           = 150
k           = 100
f           = 100.
distr_type  = 1
m           = 2
tau         = 0.8

Q_tilde_vals= []
P_X_k       = []
h_tilde     = []
NACT        = []
for l_i in range(0,30):
   lev         = float(l_i+1) / 30.
   n           = int(N*lev)
   NACT.append( n )
   # Vectors to store Gibbs binary batch simulations
   X           = np.zeros((N,),dtype=np.int32)
   X           = X.copy(order='C')
   X[0:n]      = 1
   np.random.shuffle( X )
   s_not_k     = np.sum( X ) - X[ k ]
   h_tilde_v   = math.log( (1.0 + s_not_k) / (N - s_not_k) )
   q_tilde_v   = Q_tilde(s_not_k, f, distr_type, N, m, tau)
   P_X_k.append( 1. / (1. + np.exp( -(h_tilde_v+q_tilde_v) ) ) )
   h_tilde.append( h_tilde_v )
   Q_tilde_vals.append( q_tilde_v )
   
#Plot functions
DPI            = 500

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.2, hspace = 0.05)
ax2.plot( NACT , h_tilde , linestyle='solid', linewidth=10.0, color='green')
ax2.plot([],[], label='$\widetilde{h}_{i}(x)$')
ax2.set_title("Dendritic nonlinearity",fontsize=20,pad=20)
#ax1.set_xlim(xmin=0.0,xmax=20)
ax1.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlabel("COUNT(x)",fontsize=18)
ax2.set_ylabel("$\widetilde{h}_{i}(x)$",fontsize=18)
# Set the border thickness of the spines
ax2.spines['left'].set_linewidth(5)
ax2.spines['bottom'].set_linewidth(5)
ax2.spines['right'].set_linewidth(5)
ax2.spines['top'].set_linewidth(5)

#ax2.grid()
ax2.legend(prop={'size': 40},loc='upper left',handlelength=0, handletextpad=0)
# -----------------------------------------------------------------------------
ax1.plot( NACT , P_X_k , linestyle='solid', linewidth=10.0, color='green') #, label='P($x_k$ = 1 | \"other neurons\" )')
ax1.set_xlabel("COUNT(x)",fontsize=20)
ax1.set_ylabel("P($x_k$ = 1 | \"other neurons\" )",fontsize=18)
#ax1.grid()
# Set the border thickness of the spines
ax1.spines['left'].set_linewidth(5)
ax1.spines['bottom'].set_linewidth(5)
ax1.spines['right'].set_linewidth(5)
ax1.spines['top'].set_linewidth(5)
#ax1.legend(prop={'size': 16},loc='best')
fig.suptitle("",fontsize=20)
fig.savefig('./PLOT_P_COND_H_distr'+str(distr_type)+'_N'+str(N)+'_f'+str(f)+'.png', bbox_inches = 'tight')


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.2, hspace = 0.05)
ax2.plot( NACT , Q_tilde_vals , linestyle='solid', linewidth=10.0, color='green' )
ax2.plot([],[], label='$\widetilde{Q}_{i}(x;\omega)$')
ax2.set_title("Dendritic nonlinearity",fontsize=20,pad=20)
ax1.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlabel("COUNT(x)",fontsize=18)
ax2.set_ylabel("$\widetilde{Q}_{i}(x;\omega)$",fontsize=18)
#ax2.grid()
# Set the border thickness of the spines
ax2.spines['left'].set_linewidth(5)
ax2.spines['bottom'].set_linewidth(5)
ax2.spines['right'].set_linewidth(5)
ax2.spines['top'].set_linewidth(5)
ax2.legend(prop={'size': 40},loc='best',handlelength=0, handletextpad=0)
# -----------------------------------------------------------------------------
ax1.plot( NACT , P_X_k , linestyle='solid', linewidth=10.0, color='green') #, label='P($x_k$ = 1 | \"other neurons\" )')
ax1.set_xlabel("COUNT(x)",fontsize=18)
ax1.set_ylabel("P($x_k$ = 1 | \"other neurons\" )",fontsize=18)
#ax1.grid()
# Set the border thickness of the spines
ax1.spines['left'].set_linewidth(5)
ax1.spines['bottom'].set_linewidth(5)
ax1.spines['right'].set_linewidth(5)
ax1.spines['top'].set_linewidth(5)
#ax1.legend(prop={'size': 20},loc='best')
fig.suptitle("",fontsize=20)
fig.savefig('./PLOT_P_CON_Q_distr'+str(distr_type)+'_N'+str(N)+'_f'+str(f)+'.png', bbox_inches = 'tight')