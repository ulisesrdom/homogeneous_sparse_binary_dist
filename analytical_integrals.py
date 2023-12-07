import numpy as np
import math
import scipy.special as sc

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

def expected_log_SG(r, tau, f):
   integral = ((1.0 + tau*r)/tau)*math.log( f / (1.0 + tau*r) )
   integral = integral + r
   return integral

def expected_r_log_SG(r, tau, f):
   integral = ((tau*r - 1.)/tau)*expected_log_SG( r, tau, f )
   integral = integral - (0.5*r*r)
   integral = 0.5*integral
   return integral

def expected_log_simple_SG(r, tau, f):
   integral = r*math.log( 1 + tau*r )
   integral = integral - ( r - ((1./tau)*math.log(1. + tau*r)) )
   return integral

def Ei_SG(r,tau,f, gamma):
   integral = gamma*r + r + ((1. + tau*r)/tau)*math.log( f / (1. + tau*r) )
   integral = integral + ( f / tau)*math.log( 1 + tau*r )
   s        = 0.0
   for k in range(2,41):
      s     = s + ( ( math.pow(f,k) * math.pow( 1 + tau*r, -k + 1 ) ) / ( k * (k-1) * tau * math.factorial(k) ))
   integral = integral - s
   return integral

def Ei_iterated_SG(r,tau,f, gamma):
   integral = (1./tau)*expected_log_SG( r,tau,f )
   integral = integral + expected_r_log_SG( r, tau, f )
   integral = integral + (f/tau)*expected_log_simple_SG(r, tau, f)
   integral = integral + (0.5*((r*r) + (gamma*r*r)))
   integral = integral - ((f*f/(4.*tau*tau))*math.log(1. + (tau*r)))
   s        = 0.
   for k in range(3,41):
      s     = s + ( ( math.pow(f,k) * math.pow( 1 + tau*r, -k + 2 ) ) / ( k * (k-1) * (k-2) * tau*tau * math.factorial(k) ))
   integral = integral + s
   return integral

def Ei_r_SG( r, tau,f, gamma):
   integral = r * Ei_SG( r, tau, f, gamma )
   integral = integral - Ei_iterated_SG( r, tau, f, gamma )
   return integral

def exp_SG( r, tau, f ):
   integral = ((1 + tau*r)/tau) * math.exp( f / (1.0 + tau*r) - f )
   integral = integral - (((f*math.exp(-f))/tau) * sc.expi( f / (1. + tau*r) ) )
   return integral


gamma       = 0.5772156649
tau         = 0.9 #0.4
f           = 5.
Z           = 0.301028 #0.492101 # f=5, tau=0.4
r           = 1.
q_M1_1      = 0.5*((tau*r - 1.)/tau)*exp_SG(r,tau,f) + ( 0.5*(f*math.exp(-f)/tau)* Ei_SG( r, tau, f, gamma ) )
r           = 0
q_M1_0      = 0.5*((tau*r - 1.)/tau)*exp_SG(r,tau,f) + ( 0.5*(f*math.exp(-f)/tau)* Ei_SG( r, tau, f, gamma ) )
mu_R        = (q_M1_1 - q_M1_0) / Z

r           = 1.
q_1         = ((1 + tau*r*r) / tau) * exp_SG( r, tau, f )
q_1         = q_1 + ( (2*f*math.exp(-f)/tau)*Ei_r_SG( r, tau,f, gamma) )
q_1         = q_1 - (exp_SG(r,tau,f) / tau)
q_1         = q_1 - (2*q_M1_1 / tau)

r           = 0.
q_0         = ((1 + tau*r*r) / tau) * exp_SG( r, tau, f )
q_0         = q_0 + ( (2*f*math.exp(-f)/tau)*Ei_r_SG( r, tau,f, gamma) )
q_0         = q_0 - (exp_SG(r,tau,f) / tau)
q_0         = q_0 - (2*q_M1_0 / tau)
sigma_R     = (1. / (3*Z)) * ( q_1 - q_0 ) - (mu_R * mu_R)
second_M    = (1. / (3*Z)) * ( q_1 - q_0 )
print("2nd moment = {}".format(second_M))
print("mu_r = {}, sigma_R = {} (tau={}, f={})".format(mu_R, sigma_R,tau,f))

print("Entropy_SG = {}".format( entropy_SG( f, tau ) ))