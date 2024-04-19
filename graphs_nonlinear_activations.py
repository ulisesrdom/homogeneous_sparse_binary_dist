import numpy as np
import math
#import scipy.special as sc
import functions_numerical as f_nume
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

def Q_tilde(s_not_k, f, distr_type, N, m, tau,max_order):
   q_tilde = 0.0
   S1      = 0.0
   S2      = 0.0
   if max_order < 1 or max_order > N :
      print("The maximum order must be between 1 and {}.".format(N))
      return None
   
   if distr_type == 1 : # polylogarithmic
      for j in range(1,max_order+1):
         cj = 1. / math.pow(j, m)
         S1 = S1 + (math.pow(-1,j+1)*cj*math.pow( (1. + s_not_k) / float(N) ,j))
         S2 = S2 + (math.pow(-1,j+1)*cj*math.pow( (s_not_k) / float(N) ,j))
   else:                # shifted-geometric
      for j in range(1,max_order+1):
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
   for k in range(1,n+2):
      bi_coeff = 1.0
      for j in range(0,k):
         bi_coeff = bi_coeff * float(n+1-j) / float(k-j)
      S_XK_1   = S_XK_1 + (bi_coeff * theta[k-1])
   S_XK_1   = float( n + 1 ) * math.exp( S_XK_1 )
   S_XK_0   = 0.0
   for k in range(1,n+1):
      bi_coeff = 1.0
      for j in range(0,k):
         bi_coeff = bi_coeff * float(n-j) / float(k-j)
      S_XK_0   = S_XK_0 + (bi_coeff * theta[k-1])
   S_XK_0   = float( N - n ) * math.exp( S_XK_0 )
   
   P        = S_XK_1 / ( S_XK_1 + S_XK_0 )
   return P


# -----------------------------------------------------------------------------
NQ          = 500 #20
k           = int(NQ/2.)
N_f         = 5
f_vals      = [2.0*float(NQ)] #[10.,50.,100.,150.]
m           = 5
tau         = 0.8
distr_type  = 2 #1

Q2tilde_vals= []
Q3tilde_vals= []
Q4tilde_vals= []
Q5tilde_vals= []
QNtilde_vals= []
NACT        = []
for f in f_vals:
 q2_vals    = []
 q3_vals    = []
 q4_vals    = []
 q5_vals    = []
 qN_vals    = []
 for l_i in range(0,30):
    lev         = float(l_i+1) / 30.
    n           = int(NQ*lev)
    if len(Q2tilde_vals) == 0 :
       NACT.append( n )
    # Vectors to store Gibbs binary batch simulations
    X           = np.zeros((NQ,),dtype=np.int32)
    X           = X.copy(order='C')
    X[0:n]      = 1
    np.random.shuffle( X )
    s_not_k     = np.sum( X ) - X[ k ]
    q2_vals.append( Q_tilde(s_not_k, f, distr_type, NQ, m, tau,2) )
    q3_vals.append( Q_tilde(s_not_k, f, distr_type, NQ, m, tau,3) )
    q4_vals.append( Q_tilde(s_not_k, f, distr_type, NQ, m, tau,4) )
    q5_vals.append( Q_tilde(s_not_k, f, distr_type, NQ, m, tau,5) )
    qN_vals.append( Q_tilde(s_not_k, f, distr_type, NQ, m, tau,NQ) )
 Q2tilde_vals.append( q2_vals )
 Q3tilde_vals.append( q3_vals )
 Q4tilde_vals.append( q4_vals )
 Q5tilde_vals.append( q5_vals )
 QNtilde_vals.append( qN_vals )

# -----------------------------------------------------------------------------
N             = 20
N_f           = 4
f_vals        = [10., 60., 110., 160.] #[10.,20.,30.,40.]
m             = 5
tau           = 0.8
DISTRIB_TYPE  = 2#1

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
fig, (ax2,ax1) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.2, hspace = 0.05)
n_range    = np.arange(0,N)
colors     = ['tab:blue','tab:orange','tab:green','tab:red','tab:brown']

ind        = 0
for f in f_vals:
   ax1.plot( n_range, PROB_VALS[ind,:] , linestyle='dashed', linewidth=2.0, color=colors[ind%len(colors)] , label='$f='+str(f)+'$')
   ind     = ind + 1


alpha_lev  = 0.7
ax2.plot( NACT, QNtilde_vals[0] , linestyle='solid', linewidth=3.0, color='green' ,alpha=alpha_lev, label='Nth. order (original)')
ax2.plot( NACT, Q2tilde_vals[0] , linestyle='dashed', marker='.', linewidth=2.0,alpha=alpha_lev, color='red' , label='2nd. order')
ax2.plot( NACT, Q3tilde_vals[0] , linestyle='dashed', marker='x', linewidth=2.0,alpha=alpha_lev, color='gray' , label='3rd. order')
ax2.plot( NACT, Q4tilde_vals[0] , linestyle='dashed', marker='X', linewidth=2.0,alpha=alpha_lev, color='orange' , label='4th. order')
ax2.plot( NACT, Q5tilde_vals[0] , linestyle='dashed', marker='+', linewidth=2.0,alpha=alpha_lev, color='blue' , label='5th. order')

ax1.set_xticks(np.arange(0,N,2))
ax2.set_title("Nonlinearity",fontsize=20,pad=20)
ax2.set_title("",fontsize=20,pad=20)

ax2.set_xlim(xmin=0.0,xmax=NQ)
ax2.set_xlabel("n",fontsize=18)
ax2.set_ylabel("$\widetilde{Q}_{i}(x;\omega)$",fontsize=18)
ax2.grid()
ax2.legend(prop={'size': 16},loc='best')

if DISTRIB_TYPE == 1 :
   ax1.set_title("Polylogarithmic exponential distr.",fontsize=20,pad=20)
else:
   ax1.set_title("Shifted-geometric exponential distr.",fontsize=20,pad=20)
ax1.set_title("",fontsize=20,pad=20)

ax1.set_xlabel("n",fontsize=18)
ax1.set_ylabel("P( $x_i$ = 1 | \"n active\" )",fontsize=18)
ax1.grid()
ax1.legend(prop={'size': 16},loc='best')
#ax1.set_box_aspect(1)
fig.suptitle("",fontsize=20)
fig.savefig(OUT_FOLDER+'/PROB_N_p_1_ACTIVE_N'+str(N)+'_DISTR'+str(DISTRIB_TYPE)+'_Q.png', bbox_inches = 'tight')



# ----------------------------------CONDITIONAL PROBABILITY -------------------------------------------------
N           = 500 #150
k           = 100
f           = float(2*N) #100.
distr_type  = 1
m           = 2
tau         = 0.8

Q_h_tilde_v = []
P_X_k       = []
logistic_x  = []
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
   q_tilde_v   = Q_tilde(s_not_k, f, distr_type, N, m, tau,N)
   P_X_k.append( 1. / (1. + np.exp( -(h_tilde_v+q_tilde_v) ) ) )
   h_tilde.append( h_tilde_v )
   Q_h_tilde_v.append( h_tilde_v + q_tilde_v )

x_min          = min( Q_h_tilde_v )
x_max          = max( Q_h_tilde_v )
dom_x          = []
for l_i in range(0,30):
   xp          = (float(l_i+1) / 30.) * (x_max - x_min)  + x_min
   dom_x.append( xp )
   logistic_x.append( 1. / (1. + np.exp( -xp ) ) )

#Plot functions
DPI            = 500

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.2, hspace = 0.05)
ax2.plot( NACT , h_tilde , linestyle='solid', linewidth=10.0, color='green')
ax2.plot([],[], label='$\widetilde{h}_{i}(x)$')
ax2.set_title("Dendritic nonlinearity",fontsize=50,pad=50)
ax1.set_xlim(xmin=x_min,xmax=x_max)
ax2.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlabel("COUNT(x)",fontsize=40)
ax2.set_ylabel("$\widetilde{h}_{i}(x)$",fontsize=40)
# Set the border thickness of the spines
ax2.spines['left'].set_linewidth(5)
ax2.spines['bottom'].set_linewidth(5)
ax2.spines['right'].set_linewidth(5)
ax2.spines['top'].set_linewidth(5)

#ax2.grid()
ax2.legend(prop={'size': 40},loc='upper left',handlelength=0, handletextpad=0)
# -----------------------------------------------------------------------------
ax1.plot( dom_x , logistic_x , linestyle='solid', linewidth=10.0, color='green')
ax1.set_xlabel("dendritic input",fontsize=40)
ax1.set_ylabel("Firing rate",fontsize=40)
ax1.set_title("Somatic nonlinearity",fontsize=50,pad=50)
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
ax2.plot( NACT , Q_h_tilde_v , linestyle='solid', linewidth=10.0, color='green' )
ax2.plot([],[], label='$\widetilde{h}_{i}(x) + \widetilde{Q}_{i}(x;\omega)$')
ax2.set_title("Dendritic nonlinearity",fontsize=50,pad=50)
ax1.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlim(xmin=0.0,xmax=N)
ax2.set_xlabel("synaptic inputs",fontsize=40)
ax2.set_ylabel("Dendritic output",fontsize=40)
#ax2.grid()
# Set the border thickness of the spines
ax2.spines['left'].set_linewidth(5)
ax2.spines['bottom'].set_linewidth(5)
ax2.spines['right'].set_linewidth(5)
ax2.spines['top'].set_linewidth(5)
ax2.legend(prop={'size': 40},loc='best',handlelength=0, handletextpad=0)
# -----------------------------------------------------------------------------
ax1.plot( NACT , P_X_k , linestyle='solid', linewidth=10.0, color='green') #, label='P($x_k$ = 1 | \"other neurons\" )')
ax1.set_xlabel("synaptic inputs",fontsize=40)
ax1.set_ylabel("Firing rate",fontsize=40)
ax1.set_title("Activation function",fontsize=50,pad=50)
#ax1.grid()
# Set the border thickness of the spines
ax1.spines['left'].set_linewidth(5)
ax1.spines['bottom'].set_linewidth(5)
ax1.spines['right'].set_linewidth(5)
ax1.spines['top'].set_linewidth(5)
#ax1.legend(prop={'size': 20},loc='best')
fig.suptitle("",fontsize=20)
fig.savefig('./PLOT_P_CON_Q_H_distr'+str(distr_type)+'_N'+str(N)+'_f'+str(f)+'.png', bbox_inches = 'tight')