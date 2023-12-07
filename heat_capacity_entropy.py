import numpy as np
import math
import scipy.special as sc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

# Function to compute the heat capacity for the log-modulated distribution
# at a specific value of the f parameter.
def heat_capacity_LM( f, s ):
   if f == 1.0 :
      return np.nan #0.0
   
   temp1   = math.pow(1.0,-(f-1.0)) - math.pow(2.0,-(f-1.0))
   temp2   = f - 1.0
   temp3   = math.pow( 2.0, -temp2 ) * math.log(2.0)
   Z       = temp1 / ( temp2 )
   dZ_df   = temp3 / ( temp2 )
   dZ_df   = dZ_df - ( temp1 / ( temp2*temp2 ) )
   d2Z_df2 = -temp3*math.log(2.0) / temp2
   d2Z_df2 = d2Z_df2 - ( 2.0 * temp3 / (temp2*temp2))
   d2Z_df2 = d2Z_df2 + ( 2.0 * temp1 / (temp2*temp2*temp2) )
   C_LM    = f*f * d2Z_df2 / Z
   C_LM    = C_LM - (  ((f*f) / (Z*Z)) * ( dZ_df * dZ_df )  )
   return s*C_LM

# Function to compute the entropy for the log-modulated distribution
# at a specific value of the f parameter.
def entropy_LM( f ):
   if f == 1.0 :
      Z       = math.log( 2.0 )
      H_LM    = 0.5 * math.log( 2.0 )
   else :
      temp1   = math.pow(1.0, (f-1.0))
      temp2   = math.pow(2.0, (f-1.0))
      temp3   = math.pow(1.0,-(f-1.0)) - math.pow(2.0,-(f-1.0))
      temp4   = f - 1.0
      Z       = temp3 / temp4
      H_LM    = (f / temp3) * ( (1.0 / (temp4*temp1)) - (math.log(2.0) / temp2) - (1.0 / (temp4*temp2)) )
   H_LM       = H_LM + math.log(Z)
   return H_LM
   
# Function to compute the heat capacity for the shifted geometric distribution
# at a specific value of the f and tau parameters.
def heat_capacity_SG( f, tau, s ):
   temp1   = 1.0 + tau
   temp2   = f / temp1
   temp3   = sc.expi( f ) - sc.expi( temp2 )
   Z       = (temp1 / tau) * math.exp( temp2 - f )
   Z       = Z + ( (f*math.exp(-f) / tau) * temp3 )
   Z       = Z - (1.0 / tau)
   dZ_df   = -(temp1 / tau) * math.exp( temp2 - f )
   dZ_df   = dZ_df + (1.0 / tau)
   dZ_df   = dZ_df + (  ((1.0 - f) / tau) * math.exp(-f) * temp3  )
   d2Z_df2 = ( (1.0+tau-(1.0/f)) / tau) * math.exp( temp2 - f )
   d2Z_df2 = d2Z_df2 + ( (1.0 - f) / (f*tau) )
   d2Z_df2 = d2Z_df2 + ( (math.exp(-f) * (f-2.0) / tau) * temp3 )
   
   #C_SG    = f*f * d2Z_df2 / Z
   #C_SG    = C_SG - (  ((f*f) / (Z*Z)) * ( dZ_df * dZ_df )  )
   #nume    = f*f * d2Z_df2 * Z - (  (f*f) * ( dZ_df * dZ_df )  )
   nume    = Z*f / tau
   #print("---Z*f/tau = {}".format(nume))
   part2   = -(Z*f/tau)*math.exp( temp2 - f )
   part2   = part2 - (( (f*math.exp(-f) / tau) * temp3 )**2)
   #print("---rest = {}".format(part2))
   nume    = nume -(Z*f/tau)*math.exp( temp2 - f )
   nume    = nume - (( (f*math.exp(-f) / tau) * temp3 )**2)
   #arg1    = f*( (1. / (1+tau)) - 1.0 )
   #EXP_I   = temp3 #sc.expi( f ) - sc.expi( f / (1+tau) )
   #nume    = -((1.0 + tau) / (tau*tau))*f*math.exp( 2*arg1 )
   #nume    = nume + ( ((2.0+tau)/(tau*tau))*f*math.exp(arg1) )
   #nume    = nume - (1.0 / (tau*tau))*f*f*math.exp(arg1-f)
   #nume    = nume - ( (f*f / (tau*tau))*math.exp(-2*f)*EXP_I*EXP_I )
   #nume    = nume + ( (2.0*(f*f)/ (tau*tau))*math.exp(-f)*EXP_I )
   #nume    = nume - ( ((f*f*f)/(tau*tau))*math.exp(-f)*EXP_I )
   #nume    = nume + (f*(f-1) / (tau*tau))
   denom    = Z*Z
   C_SG    = nume / denom
   #print(" f = {}, nume = {}, denom = {}, L = {}".format(f,nume,denom,C_SG))
   return s*C_SG


# Function to compute the entropy for the shifted geometric distribution
# at a specific value of the f and tau parameters.
def entropy_SG( f, tau ):
   temp1   = 1.0 + tau
   temp2   = f / temp1
   temp3   = sc.expi( f ) - sc.expi( temp2 )
   Z       = (temp1 / tau) * math.exp( temp2 - f )
   Z       = Z + ( (f*math.exp(-f) / tau) * temp3 )
   Z       = Z - (1.0 / tau)
   
   H_SG    = math.log( Z )
   H_SG    = H_SG - ( ( f*math.exp(-f) * temp3 ) / (tau *  Z) ) + f
   return H_SG

max_f     = 3#100.0 #25.0 #5.0 #2.5#20.0 #2.35 #10.0
min_f     = 0.0 #1.1#1.4
delt      = 0.04
tau       = 0.7
DPI       = 500
OUT_FOLDER= '.'
f_vals    = np.arange( min_f + delt, max_f + delt , delt )
#print("{}".format(f_vals))
C_SG      = np.zeros((f_vals.shape[0],),dtype=np.float32)
H_SG      = np.zeros((f_vals.shape[0],),dtype=np.float32)
C_LM      = np.zeros((f_vals.shape[0],),dtype=np.float32)
H_LM      = np.zeros((f_vals.shape[0],),dtype=np.float32)
ind       = 0
for f in f_vals:
   C_SG[ ind ] = heat_capacity_SG( f, tau, 1.0 )
   H_SG[ ind ] = entropy_SG( f, tau )
   C_LM[ ind ] = heat_capacity_LM( f, 1.0 )
   H_LM[ ind ] = entropy_LM( f )
   ind      = ind + 1


#Plot results
# -----------------------------------------------------------------------------
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6),dpi=DPI)
fig.subplots_adjust(top=0.9, bottom = 0.1, right=0.9, left=0.1, wspace = 0.1, hspace = 0.05)
#fig  = plt.figure(figsize=(6,5),dpi=DPI)
#fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
#ax1  = fig.add_subplot(1,1,1)
ax2.plot( f_vals , C_SG , linestyle='dashed', linewidth=2.0, color='cornflowerblue', label='Shifted-geometric $\\tau='+str(tau)+'$')
#ax2.plot( f_vals , C_LM , linestyle='dashed', linewidth=2.0, color='green', label='Log-modulated')
# single point at discontinuity where heat capacity is zero
ax2.plot( 1.0 , 0.041 , linestyle='dashed', marker='o', markersize=12, linewidth=3.0,  mfc='none', color='green', label='')
ax2.set_title("Heat capacity",fontsize=20,pad=20)
#ax1.set_xscale("log")
#ax1.set_yscale("log")
#ax1.set_xlim(xmin=0.03,xmax=10)
#ax1.set_ylim(ymin=0.03,ymax=10)
ax2.set_xlabel("$f$",fontsize=18)
ax2.set_ylabel("C( $f$ )",fontsize=18)
ax2.grid()
ax2.legend(prop={'size': 16},loc='best')
#ax1.set_box_aspect(1)


# -----------------------------------------------------------------------------

ax1.plot( f_vals , H_SG , linestyle='dashed', linewidth=2.0, color='cornflowerblue', label='Shifted-geometric $\\tau='+str(tau)+'$')
ax1.plot( f_vals , H_LM , linestyle='dashed', linewidth=2.0, color='green', label='Log-modulated')
ax1.set_title("Entropy",fontsize=20,pad=20)
ax1.set_xlabel("$f$",fontsize=18)
ax1.set_ylabel("H( $f$ )",fontsize=18)
ax1.grid()
ax1.legend(prop={'size': 16},loc='best')
#ax1.set_box_aspect(1)
fig.suptitle("",fontsize=20)
fig.savefig(OUT_FOLDER+'/ENTROPY_HEAT_CAPACITY_SG_TAU_'+str(tau)+'_LM.png', bbox_inches = 'tight')


# Numerically find maximum
# -------------------------------------------------------------------------

# Log-modulated model
# ----------------------------
#res = minimize(-heat_capacity_LM, 5.0, method='BFGS', jac=rosen_der, options={'disp': True})
res = minimize( heat_capacity_LM, 5.0, args=(-1.0), method='BFGS', options={'disp': True})
print("{}".format(res.x))

# Shifted-geometric model
# ----------------------------
res = minimize( heat_capacity_SG, 5.0, args=(tau,-1.0), method='BFGS', options={'disp': True})
print("{}".format(res.x))
