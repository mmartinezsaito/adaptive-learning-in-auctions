#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, os, csv, pdb
import pandas as pd
import scipy as sp
"""http://docs.scipy.org/doc/scipy-0.15.1/reference/optimize.html
   http://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html"""
from scipy.optimize import minimize
from scipy import stats
nan = '\x00\x00\x00\x00\x00\x00\xf8\x7f'
#import theano
import sklearn as skl
#import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from statsmodels.base.model import GenericLikelihoodModel
import functools
import numdifftools as nd
import copy
import random
import pickle
#from pybrain.rl.environments.mazes import Maze, MDPMazeTask
#from pybrain.rl.learners.valuebased import ActionValueTable
#from pybrain.rl.agents import LearningAgent
#from pybrain.rl.learners import Q, SARSA
#from pybrain.rl.experiments import Experiment
#from pybrain.rl.environments import Task



project = "econoshyuka" # "neuroshyuka" or "econoshyuka"

initbid = {}
if project == "neuroshyuka":
    fn = os.path.dirname(os.path.dirname(os.getcwd())) + os.path.sep + 'Data' + os.path.sep + 'shyuka.csv'
elif project == "econoshyuka":
    fn = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/BargainingStudyControl_EYA/logs/econoshyuka.csv'

with open(fn, 'r') as csvfile:
    fr = csv.reader(csvfile)  
    df = [row for row in fr]

D = pd.read_csv(fn)
S = D['sid'].unique()
if project == 'econoshyuka':
    only_cond = {'d'} #  bcd{'b','c','d'}
    D = D.drop([i for i in D.index if D.ix[i,'cond'] not in only_cond])
    D2 = D
    if only_cond == {'c'}:
        fnp = os.path.dirname(os.path.dirname(os.getcwd())) + os.path.sep + 'Data' + os.path.sep + 'PrerecordedData.csv'
        with open(fnp, 'r') as csvfile:
            fr = csv.reader(csvfile)  
            df = [row for row in fr]
        Dp = pd.read_csv(fnp, header = None)      
        opp_role = {'seller_NC': list(range(24)), 'seller_SC': list(range(24,48)), 'buyer_BC': list(range(48,72)), 'seller_BC': list(range(72,96))} 
else:
    only_cond = {''};
    S2 = D.sid.unique()[18:]
    D2 = D[D['sid'].isin(S2)]

hl = D.columns.values.tolist()
print(D)

r_m = sp.mean(D.ix[:, 'profit'])
ns = D.shape[0]
a = 0.5 # learning rate
b = 1 # inverse temperature
Bbins = 101

if project == "neuroshyuka":
    initbid['BC'] = 6.554348
    initbid['NC'] = 5.131915
    initbid['SC'] = 4.959574
elif project == "econoshyuka":
    initbid['BC'] = D[(D['snb']=='BC') & (D['block_n']==1)]['bid'].mean() 
    initbid['NC'] = D[(D['snb']=='NC') & (D['block_n']==1)]['bid'].mean() 
    initbid['SC'] = D[(D['snb']=='SC') & (D['block_n']==1)]['bid'].mean()
print(initbid)



####################################
###  Various function utilities  ###
####################################

def round2(n):
    return round(10 * n) / 10

def buyer_profit(bid):
    return 10 - bid

def num_hessian(x0, cost_function, epsilon = 1.e-5, linear_approx = False, *args):
    """ A numerical approximation to the Hessian matrix of cost 
    function at location x0 (hopefully, the minimum) """
    # The next line calculates an approximation to the first derivative
    f1 = sp.optimize.approx_fprime(x0, cost_function, *args) 
    # This is a linear approximation, efficient if cost function is linear
    if linear_approx:
        f1 = sp.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = sp.zeros ((n, n))
    # The next loop fill in the matrix
    xx = x0
    for j in xrange(n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = sp.optimize.approx_fprime(x0, cost_function, *args) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian

# Root finding algorithms
def srf(x, bdu, bdv, bdmax = 10):
    return bdmax - beta_dist_maxB(bdu, bdv, x) #x * (1 - bdmax / beta_dist_maxB(bdu, bdv, x)) 
def newtonraphson_secant(x0, args):
    """ Newton-Raphson and Secant methods: find a zero of a scalar function
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton 
     Find a zero of the function func given a nearby starting point x0. 
     The Newton-Raphson method is used if the derivative fprime of func is provided, otherwise the secant method is used. 
     If the second order derivate fprime2 of func is provided, parabolic Halley’s method is used.
     The convergence rate of the Newton-Raphson method is quadratic, the Halley method is cubic, and the secant method is sub-quadratic. 
     This means that if the function is well behaved the actual error in the estimated zero is approximately 
      the square (cube for Halley) of the requested tolerance up to roundoff error. 
     However, the stopping criterion used here is the step size and there is no guarantee that a zero has been found. 
     Consequently the result should be verified. Safer algorithms are brentq, brenth, ridder, and bisect, but they all 
      require that the root first be bracketed in an interval where the function changes sign. The brentq algorithm is 
      recommended for general use in one dimensional problems when such an interval has been found.
    """
    func = srf
    fprime = None # secant is used if this is None
    if args[0:2] == (5, 1): 
        return args[2]
    zero = sp.optimize.newton(func, x0, fprime = fprime, args = args)
    return zero
def brent(a, b, args):
    """ Brent's method: find a root of a scalar function in a given interval
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq
     Return float, a zero of f between a and b. f must be a continuous function, and [a,b] must be a sign changing interval.
     Uses the classic Brent (1973) method to find a zero of the function f on the sign changing interval [a , b]. 
      Generally considered the best of the rootfinding routines here. It is a safe version of the secant method that uses 
      inverse quadratic extrapolation. Brent’s method combines root bracketing, interval bisection, and inverse quadratic 
      interpolation. It is sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973) claims convergence is 
      guaranteed for functions computable within [a,b].
    """
    f = srf
    if args[0:2] == (5, 1):
        return args[2]
    x0, r = sp.optimize.brentq(f, a, b, args = args, disp= True, full_output = True)   
    return x0



##############################
###  Value initialization  ###
##############################

# This will assess the relative importance of a purported game-theoretic analysis

if Bbins == 101:
    B = sp.linspace(0, 10, 101)
elif Bbins == 11:
    B = sp.linspace(0, 10, 11)

# Naive CHL-0
initQ_CH0 = sp.array([float(buyer_profit(i)) for i in B]) #initQd_CH0 = {round2(i): buyer_profit(i) for i in B} 

# Naive CHL-0 * logisticP(reject)
def logisticB(loc, scale):
    logistic_frz = sp.stats.logistic(loc = loc, scale = scale)
    return sp.array([logistic_frz.pdf(i / 10) for i in B])[::-1]
def CH0lPrB(loc):
    scale = 1
    arr_lPr = logisticB(loc, scale)
    arr_CH0 = sp.array([float(buyer_profit(i)) for i in B]) #initQd_CH0 = {round2(i): buyer_profit(i) for i in B} 
    return arr_lPr * arr_CH0
thr = 5
scale = 1
initQ_CH0lPr = CH0lPrB(thr) 

# Uniform initialization to domain mean
def unifB(q):
     return sp.array([float(q) for i in B]) #initQd_unif = {round2(i): q0 for i in B}
q0 = 5
initQ_unif = unifB(q0)

# Parametric Beta distribution initialization
def ab2uv(a, b):
    return a / float(a + b), -sp.log(a + b) 
def uv2ab(u, v):
    return sp.exp(-v) * u, sp.exp(-v) * (1 - u) 
"""https://en.wikipedia.org/wiki/List_of_probability_distributions"""
def beta_distB(u, v, c, isab = False):
    #u = a / (a + b) # mean 
    #v = -sp.log(a + b) # volatility, in log-space, the natural space for a variance parameter
    if not isab:
        u *= 0.1 
        v = -v * sp.log(2) 
        a, b = uv2ab(u, v)
    else:
        a, b = u, v
    sc = 10
    beta_frz = sp.stats.beta(a, b, loc = 0 - 10**-4, scale = sc + 2*10**-4)
    return sp.array([c * sc * beta_frz.pdf(i) for i in B])
def gamma(z):
    return sp.special.gamma(z)
def digamma(z): # logarithmic derivative of the gamma function
    return sp.special.psi(z)
def beta_distB_da(a, b, c):
    return beta_distB(a, b, c) * (-digamma(a + b) + digamma(a))
def beta_distB_db(a, b, c):
    return beta_distB(a, b, c) * (-digamma(a + b) + digamma(b))
def beta_dist_mode(a, b):
    if b < 1:
        return 1
    if a < 1:
        return 0
    return float(a - 1) / (a + b - 2)
def beta_dist_max(a, b):
    return sp.stats.beta.pdf(beta_dist_mode(a, b), a = a, b = b, loc = 0 - 10**-4, scale = 1 + 2*10**-4)
def beta_dist_maxB(u, v, c):
    u *= 0.1 
    v = -v * sp.log(2) 
    a, b = uv2ab(u, v)
    return c * sp.stats.beta.pdf(beta_dist_mode(a, b), a = a, b = b, loc = 0 - 10**-4, scale = 1 + 2*10**-4)

# plot beta distribution
def bdB(u, v, c, ax = None, isab = False):
    #u = a / (a + b) # mean 
    #v = -sp.log(a + b) # volatility, in log-space, the natural space for a variance parameter
    if not isab:
        u *= 0.1 
        v = -v * sp.log(2)
        a, b = uv2ab(u, v)
    else: 
        a, b = u, v
    sc = 10
    prior = functools.partial(sp.stats.beta.pdf, a = a, b = b, scale = sc)
    x = sp.linspace(0, 10, num = 200)
    plt.title("%s * Beta(%s, %s) scaled %sx%s "% (c, a, b, sc, sc))
    c *= sc
    y = c * prior(x) # c * sp.array(map(prior, x))
    plt.plot(x, y)
    if not ax:
        plt.show()
        

# Q (delta rule)
initQ = sp.copy(initQ_unif) # for NumPy! nitQ[:] = initQ_unif 
markets = {'BC', 'NC', 'SC'}
Q = dict(BC = sp.copy(initQ), 
         NC = sp.copy(initQ),
         SC = sp.copy(initQ))



###############################
###  Some benchmark models  ###
###############################

# Null model
def null_agent():
    return 1.0 / len(B)
   
# Game theoretic Nash equilibrium double auction solution  
"""Policy: choose always 9.9 in BC, choose anything in the other conditions"""
def gt_NE_agent(m):
    N = len(B)
    S = 10.0 / N 
    if m == 'BC':
        p = 1
        q = N - S     
    else:
        p = 1.0 / N 
        q = round2(sp.random.randint(N) * S)
    return p, q

# Game theoretic Bayesian Nash equilibrum double auction solution  
""" Chatterjee K, Samuelson W (1983) Bargaining under incomplete information """
""" Felli L (2002) Bilateral asymmetric information """
"""  assumes that both the buyer and the seller believe that the valuation of 
     the opponent (vs, vb respectively) is uniformly distributed on [0, len(B)]
     this is a ChInf model"""
def gt_BayesianNE_agent(m, vb = 1, vs = 1):
    N = len(B)
    S = 10.0 / len(B)
    if   m == 'BC':
        p = 1
        qb = N - S     
    elif m == 'NC' or m == 'BC':
        p = 1
        qb = (2.0/3 * vs + 1.0/12) * N
        qs = (2.0/3 * vb + 1.0/4) * N
    return p, qb    



#####################################
###  Cognitive Hierarchy Level 1  ###
#####################################
# assumes buyer tries to maximize profit while treating sellers as non-agents, but 'nature'
# estimates seller behavior, including maybe reserve price density, as part of nature


## Policy / observation model

def partition_func(b, L):
    return sum([sp.exp(b * i) for i in L])

def boltzmann_dist(b, E, i):
    Z = partition_func(b, E)
    return sp.exp(b * E[i]) / Z


## Learning model

# Model-free RL: value-based delta rule 
def dr_naive_avf_agent(a, b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 

    Q[bi] += a * (r - Q[bi])
    return p, Q

def ql_naive_agent(a, b, Q, bi, r):
    gamma = 0 # if zero, same as dr_naive, since it's a stage game: single-shot 
    p = boltzmann_dist(b, Q, bi) 
    Q[bi] += a * (r + gamma * max(Q) - Q[bi])
    return p, Q


# Model-free RL: policy-based updating
def pu_naive_avf_agent(b, A, bi, r):
    p = boltzmann_dist(b, A, bi) 

    A[bi] += r - r_m 
    return p, A

# Model-based RL: rule understanding: acceptance if, seller reserve price > bid
# Simple delta rule batch asymmetrical updating
def dr_unilatupd_agent(a, b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 

    if r == 0: # rejected
        for j in range(len(Q)):
            if j <= bi: 
                Q[j] += a * (0 - Q[j]) # r=0 always
            else:
                pass
    else:  # accepted
        for j in range(len(Q)):
            if j >= bi: 
                Q[j] += a * (initQ_CH0[j] - Q[j]) 
            else:
                pass
    return p, Q

def pu_unilatupd_agent(b, Q, bi, r):
    p = boltzmann_dist(b, Q, bi) 
    du = [r - r_m for q in Q]
    du_acc = [initQ_CH0[i] - r_m for i in range(len(Q))]
    if r == 0: # rejected
        for j in range(len(Q)):
            if j <= bi: 
                Q[j] += du[j]
            else:
                pass
    else:  # accepted
        for j in range(len(Q)):
            if j >= bi: 
                Q[j] += du_acc[j]
            else:
                pass
    return p, Q


### Kernel density estimation of opponent's choice pdfs ###

def kdeag_uf(Ks1 = None, Ks2 = None, Kb2 = None):
    """ computes bargainer utility function given density estimates """
    U = sp.zeros(Bbins) 
    for i in range(Bbins):
        for s1 in range(Bbins):
            if Ks2 is None and Kb2 is None:
                if B[i] < B[s1]: 
                    U[i] += 0
                else:
                    U[i] += (10 - B[i]) * Ks1[s1]
            elif Ks2 is not None and Kb2 is None:
                for s2 in range(Bbins):
                    t = min(B[s1], B[s2])
                    if B[i] < t: 
                        U[i] += 0
                    else:
                        U[i] += (10 - B[i]) * Ks1[s1] * Ks2[s2]
            elif Ks2 is None and Kb2 is not None:
                for b2 in range(Bbins):
                    t = max(B[s1], B[b2])
                    if B[i] < t: 
                        U[i] += 0
                    else:
                        U[i] += (10 - B[i]) * Ks1[s1] * Kb2[b2]
    return U


#skde = skl.neighbors.KernelDensity(bandwidth = bw, algorithm = 'auto', kernel = 'gaussian', metric = 'euclidean')
#skf = skde.fit(D.ix[i0:i, 's1_rp'].reshape(-1,1))
#plt.plot(skf.score_sample(sp.linspace(0,10,Bbins).reshape(-1,1))))
def kde_agent(beta, bid, i, m, bw = None, *args):
    """ simulates kde artificial bargainer behavior """
    if bw is None: bw = 'scott'

    i0 = D.index[0]
    if only_cond == {'d'}:
        s1 = D.ix[i0:i, 's1_rp']
        if len(s1) == 1:
            s1[i0+1] = sp.mean([s1, 10])
    else:
        s1est = args[0]
        ar = D.ix[i, 'out_bool']
        b1 = D.ix[i, 'bid']
        b1i = round2(b1)
        if len(s1est) == 1:
            s1est.append(sp.mean([s1est[0], 10]))  
        else:
            gks = stats.gaussian_kde(s1est, bw_method = bw)
            if ar:
               aestint = gks.pdf(sp.arange(0, b1i, 0.1))
               aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
            else:
               aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
               aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
            s1est.append(aest) 
        s1 = s1est
    gks = stats.gaussian_kde(s1, bw_method = bw)
    Ks1 = gks.pdf(B) 
    Ks1 /= sum(Ks1)

    if m == 'SC':
        if only_cond == {'d'}:
            s2 = D.ix[i0:i, 's2_rp'].dropna()
            if len(s2) == 1:
                s2[i+1] = sp.mean([s2[i], 10])  
        else:
            s2est = args[1]
            if len(s2est) == 1:
                s2est.append(sp.mean([s2est[0], 10]))  
            else:
                gks = stats.gaussian_kde(s2est, bw_method = bw)
                if ar:
                    aestint = gks.pdf(sp.arange(0, b1i, 0.1))
                    aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
                else:
                    aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
                    aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
                s2est.append(aest) 
            s2 = s2est
        gks = stats.gaussian_kde(s2, bw_method = bw)
        Ks2 = gks.pdf(B) 
        Ks2 /= sum(Ks2)
        U = kdeag_uf(Ks1, Ks2, None)
        s2orb2 = s2
    elif m == 'NC':
        U = kdeag_uf(Ks1, None, None)
        s2orb2 = None
    elif m == 'BC':
        if only_cond == {'d'}:
            b2 = D.ix[i0:i, 'b2_bid'].dropna()
            if len(b2) == 1:
                b2[i+1] = sp.mean([b2[i], 10])  
        else:
            b2est = args[1]
            if len(b2est) == 1:
                b2est.append(sp.mean([b2est[0], 10]))  
            else:
                gks = stats.gaussian_kde(b2est, bw_method = bw)
                if ar:
                    aestint = gks.pdf(sp.arange(0, b1i, 0.1))
                    aest = sp.dot(aestint, sp.arange(0, b1i, 0.1)) / b1i 
                else:
                    aestint = gks.pdf(sp.arange(b1i, 10, 0.1))
                    aest = sp.dot(aestint, sp.arange(b1i, 10, 0.1)) / (10-b1i) 
                b2est.append(aest) 
            b2 = b2est
        gks = stats.gaussian_kde(b2, bw_method = bw)
        Kb2 = gks.pdf(B) 
        Kb2 /= sum(Kb2)
        U = kdeag_uf(Ks1, None, Kb2)
        s2orb2 = b2
    
    opb = B[sp.argmax(U)]
    p = boltzmann_dist(beta, U, int(bid*((Bbins-1)/10)))#p = P[int(bid*((Bbins-1)/10))]
    return p, opb, s1, s2orb2, U



####################################################################
### Nudgers: bumping the bid up and down in a markovian fashion  ###
####################################################################

# Logistic probability of rejection updating
def nudger_CH0lPr_agent(nud, b, Q, bi, thr):
    p = boltzmann_dist(b, Q, bi)
    if  r == 0:
        thr += nud
    else:
        thr -= nud
    Q = CH0lPrB(thr) 
    return p, Q

# Naive gaussian nudging model
def naive_gausnudger1_agent(n_up, n_dn, sig, q, bid, r):
    #sig = (n_up + n_dn) / 2
    p = sp.stats.norm.pdf(bid, loc = q, scale = sig)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
def naive_gausnudger2_agent(n_up, n_dn, siga, sigr, q, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = q, scale = siga)
    else:
        p = sp.stats.norm.pdf(bid, loc = q, scale = sigr)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
# Improve using a leptokurtic distribution
# Naive laplacian(leptokurtic) nudging model
def naive_lepkurnudger1_agent(n_up, n_dn, sig, q, bid, r):
    p = sp.stats.laplace.pdf(bid, loc = q, scale = sig)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q
def naive_lepkurnudger2_agent(n_up, n_dn, siga, sigr, q, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = q, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = q, scale = sigr)
    if  r == 0:
        q += n_up
    else:
        q -= n_dn
    return p, q



###  Linear Kalman filter  ###

#  mean-tracking rule for the optimal bid: this is not an action-value function 
#  needs a prior for mu, sb, s0 
def kalman_gausnudger1_agent(mu, sb, s0, bid, r):
    p = sp.stats.norm.pdf(bid, loc = mu, scale = sb)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
    else:
        d = bid - mu    # innovation or measurement residual
    a = sb**2 / (sb**2 + s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    sb *= sp.sqrt(1 - a) # updated estimate covariance
    return p, mu, sb
def kalman_gausnudger2_agent(mu, sba, sbr, s0, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sba)
    else:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sbr)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
        a = sbr**2 / (sbr**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sbr *= sp.sqrt(1 - a) # updated estimate covariance
    else:
        d = bid - mu    # innovation or measurement residual
        a = sba**2 / (sba**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sba *= sp.sqrt(1 - a) # updated estimate covariance
    a = (sba**2 + sbr**2) / (sba**2 + sbr**2 + 2*s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    return p, mu, sba, sbr
# Leptokurtic kalman nudgers
def kalman_lepkurnudger1_agent(mu, sb, s0, bid, r):
    p = sp.stats.laplace.pdf(bid, loc = mu, scale = sb)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
    else:
        d = bid - mu    # innovation or measurement residual
    a = sb**2 / (sb**2 + s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    sb *= sp.sqrt(1 - a) # updated estimate covariance
    return p, mu, sb
def kalman_lepkurnudger2_agent(mu, sba, sbr, s0, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sba)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sbr)
    if r == 0:
        d = 0 # + sb**2 / (sb**2 + s0**2) 
        a = sbr**2 / (sbr**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sbr *= sp.sqrt(1 - a) # updated estimate covariance
    else:
        d = bid - mu    # innovation or measurement residual
        a = sba**2 / (sba**2 + s0**2) # optimal Kalman gain (with innovation covariance)
        sba *= sp.sqrt(1 - a) # updated estimate covariance
    a = (sba**2 + sbr**2) / (sba**2 + sbr**2 + 2*s0**2) # optimal Kalman gain (with innovation covariance)
    mu += a * d # updated state estimate
    return p, mu, sba, sbr


### Delta rule nudger ###

def dr_gausnudger1_agent(a, sig, mu, bid, r): # LOUSY AGENT, BE ASHAMED
    p = sp.stats.norm.pdf(bid, loc = mu, scale = sig)
    if r == 0: 
        d = 0 # + a  
    else:
        d = bid - mu  # innovation or measurement residual   
    mu += a * d # updated state estimate
    return p, mu
def dr_gausnudger2_agent(a, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.norm.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = 0 ## what i lost, or my counterfactual gain
    else:
        d = bid-mu   #   
    mu += a * d # updated state estimate
    #mu = min(10, max(0, mu))
    return p, mu
# Leptokurtic dr nudgers
def dr_lepkurnudger1_agent(a, sig, mu, bid, r): 
    p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig)
    if r == 0: 
        d = 0   
    else:
        d = bid - mu  # innovation or measurement residual   
    mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger2_agent(a, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = 0 ## what i lost, or my counterfactual gain
    else:
        d = bid-mu   #   
    mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger31_agent(nd, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = +nu ## what i lost, or my counterfactual gain
    else:
        d = (bid - mu) -nd  #   
    mu += d # updated state estimate
    return p, mu
def dr_lepkurnudger32_agent(a, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        d = +nu ## what i lost, or my counterfactual gain
        mu += d
    else:
        d = (bid - mu)  #   
        mu += a*d  # updated state estimate
    return p, mu
def dr_lepkurnudger33_agent(aa, ar, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    d = (bid - mu)     
    if r == 0:
        mu += ar*d # updated state estimate
    else:
        mu += aa*d # updated state estimate
    return p, mu
def dr_lepkurnudger4_agent(a, nd, nu, siga, sigr, mu, bid, r, prevacc):
    if prevacc:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga)
    else:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr)
    if r == 0:
        mu += nu # updated state estimate
    else:
        d = (bid - mu)     
        mu += a * d -nd # updated state estimate
    return p, mu
def dr_lepkurnudger5_agent(a, nd, nu, siga, sigr, sign, raitc, mu, bid, r, prevacc): # OVERFITS? 
    # the pdf of a sum of random variables is the convolution of their corresponding pdfs respectively
    #  https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions 
    # i will use a trick with raitc to implement tradeoff sigar and sign
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    if r == 0:
        mu += nu # updated state estimate
    else:
        d = (bid - mu)     
        mu += a * d -nd # updated state estimate
    return p, mu
def dr_lepkurnudger6_agent(a, siga, sigr, sign, raitc, mu, bid, r, prevacc): # OVERFITS? 
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    d = (bid - mu)     
    if r > 0:
        mu += a * d # updated state estimate
    return p, mu
def dr_lepkurnudger7_agent(a, siga, sigr, sign, raitc, mu, bid, r, prevacc): 
    if   prevacc and bid <= mu: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = siga) * 2 * (1 - raitc) 
    elif prevacc and bid > mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc 
    elif not prevacc and bid >= mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sigr) * 2 * (1 - raitc)
    elif not prevacc and bid < mu:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sign) * 2 * raitc
    mu += a * (bid - mu) # updated state estimate
    return p, mu
def dr_fb1_lepkurnudger6_agent(a, sig_lt, sig_e, sig_gt, raitc, mu, bid, r, prevcob, cob): 
    if   bid > prevcob: 
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_gt) * 2 * raitc 
    elif bid == prevcob:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_e) * 2 * raitc 
    elif bid < prevcob:
        p = sp.stats.laplace.pdf(bid, loc = mu, scale = sig_lt) * 2 * (1 - raitc)
    mu += a * (cob - mu) # updated state estimate
    return p, mu



####################
###  Estimation  ###
####################

# Explicit computation of L, yoked parameters
class llhf_ffx_explicit:  # numerical log-likelihood  function
    def __init__(self, mms_a, mms_b):
        if not mms_a:
            mms_a = 0, 1, 10**-2
        if not mms_b:
            mms_b = 10**-2, 1, 10**-2
        self.mms_a = mms_a
        self.mms_b = mms_b
        self.dom_a = sp.arange(*mms_a)
        self.dom_b = sp.arange(*mms_b)
        self.f = sp.zeros((len(self.dom_a), len(self.dom_b)))
         # self.f = {(a, b): 0 for a, b in zip(self.dom_a, self.dom_b)}
    def update(self, a, b, logp):
        x_a = round((a - self.mms_a[0]) / self.mms_a[2]) 
        x_b = round((b - self.mms_b[0]) / self.mms_b[2])
        self.f[x_a, x_b] += logp

def mle_ffx_yoked_explicit():
    initQ = sp.copy(initQ)
    mms_a = (0, 1.1, 0.25)
    mms_b = (0.1, 2.1, 0.5)
    Ln = llhf_explicit(mms_a, mms_b) 
    for a in Ln.dom_a:
        for b in Ln.dom_b:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    Q[m] = sp.copy(initQ)
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    if Bbins == 101: 
                        ind = int(bid * 10) 
                    elif Bbins == 11:
                        ind = int(round(bid)) 
                    p, Q[m] = pu_naive_avf_agent(b, Q[m], ind, r)
                    #p, Q[m] = dr_naive_avf_agent(a, b, Q[m], ind, r)
                    Ln.update(a, b, sp.log10(p))
                    print(row, a, b, m, bid, r, Q[m], sp.log10(p))

    Lmax = sp.amax(Ln.f)
    mlep = sp.where(Ln.f == Lmax) #mlep = sp.argmax(Ln.f) 
    mle_a, mle_b = mlep[0] * Ln.mms_a[2] + Ln.mms_a[0], mlep[1] * Ln.mms_b[2] + Ln.mms_b[0]
    
    fig, axa = plt.subplots(2, 1)

    plt.subplot(211) # fig = plt.figure(); axs = fig.add_subplot(); axs.plot()
    levels = sp.linspace(sp.amin(-Ln.f), sp.amax(-Ln.f), 20)
    X, Y = sp.meshgrid(Ln.dom_a, Ln.dom_b)
    plt.contourf(X, Y, -Ln.f.transpose(), levels = levels, alpha = .5, cmap = plt.cm.gray) 
    plt.colorbar(format = '%.2f') 
    isoLn = plt.contour(X, Y, -Ln.f.transpose(), levels = levels, colors='black', linewidth=.5)
    plt.clabel(isoLn, inline=1, fontsize=10)
    plt.annotate('Lmax', xy = (min(mle_a), min(mle_b)), xytext = (0.1, 0.1), arrowprops = dict(facecolor='b', width=0.01, shrink=0.02))

    plt.title('NLL; Lmax=L({0},{1})={2}'.format(mle_a, mle_b, Lmax))
    plt.xlabel('a')
    plt.ylabel('b')
    #plt.xticks(())
    #plt.yticks(())
    plt.legend()
    
    ax3 = fig.add_subplot(212, projection='3d')
    ax3.plot_wireframe(X, Y, -Ln.f.transpose()) #plot_surface

    pdb.set_trace()
    plt.show() # or plt.savefig(); plt.close() 

    return mle_a, mle_b

# Bid nudging
def mle_ffx_yoked_explicit_nudging(initq = 5):
    q = dict(BC = initq,
             NC = initq,
             SC = initq)
    mms_nup = (0, 1.5, 0.5)
    mms_ndn = (0, 1.5, 0.5)
    dom_nup = sp.arange(*mms_nup)
    dom_ndn = sp.arange(*mms_ndn)
    L = sp.zeros((len(dom_nup), len(dom_ndn)))
    for nu in dom_nup:
        for nd in dom_ndn:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    q[m] = initq
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    p, q[m] = naive_gausnudger_agent(nu, nd, q[m], bid, r)
                    L[nu, nd] += sp.log10(p) 
                    print(row, nu, nd, m, bid, r, q[m], sp.log10(p))
    Lmax = sp.amax(L)
    mlep = sp.where(L == Lmax) #mlep = sp.argmax(Ln.f) 
    mle_nup, mle_ndn = mlep[0] * mms_nup[2] + mms_nup[0], mlep[1] * mms_ndn[2] + mms_ndn[0]

    fig = plt.figure(1)
    plt.subplot(111) 
    X, Y = sp.meshgrid(dom_nup, dom_ndn)
    plt.contourf(X, Y, -L.transpose(), 8, alpha = .8, cmap = plt.cm.hot) 
    isoLn = plt.contour(X, Y, -L.transpose(), 8, colors='black', linewidth=.5)
    plt.clabel(isoLn, inline=1, fontsize=10)
    plt.annotate('Lmax', xy = mlep, xytext = (0.1, 1), arrowprops = dict(facecolor='b', shrink=0.02))

    plt.title('NLL')
    plt.xlabel('nup')
    plt.ylabel('ndn')
    plt.legend()

    ax3 = fig.add_subplot(212, projection='3d')
    ax3.plot_wireframe(X, Y, -L.transpose())
    
    pdb.set_trace()
    plt.show() 


## Local extrema search in L, yoked parameters

# Objective functions: Fixed effects likelihood functions
""" For population-level questions, treating parameters as fixed effects and thereby
conflating within- and between-subject variability can lead to serious problems 
such as overstating the true significance of results"""

def nll_ffx_dr_avf(params, D = D2):
    initQ = {}
    if len(params) == 2:
        fn = 'behvars_dr_avf4.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda = initbid
        bdb, bdc = 2, 4
        initQ = {m: beta_distB(initbid[m], bdb, bdc) for m in markets}
    elif len(params) == 5:
        fn = 'behvars_dr_avf1.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda, bdb, bdc = params[2:5]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 9 and 1==0:
        fn = 'behvars_dr_avf2.csv'
        a = {i[1]: params[0:6:2][i[0]] for i in enumerate(markets)}
        b = {i[1]: params[1:6:2][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[6:9][i[0]]) for i in enumerate(markets)}
    elif len(params) == 9:
        fn = 'behvars_dr_avf3.csv'
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bdab = params[2:8]
        use_bdmax = True 
        if not use_bdmax:
            bdc = params[8]
        elif use_bdmax:
            bdmax = params[8]
            #bdc = newtonraphson_secant(5, (bda, bdb)) 
            try: 
                bdc = min([brent(0, 10, (bdab[2*i], bdab[2*i+1], bdmax)) for i in range(3)])
            except ValueError:
                return 10**10 
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    nt = len(D)
    nb = max(D.ix[:, 'block_n'])
    emv, opb, rpe, ecv, clh, cob, coe, cmr, sve = [], [], [], [], [], [], [], [], []    
    writevars = True
    if writevars:
        fid = open(fn, 'w') 
        cw = csv.writer(fid, delimiter = ',')
        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve", "emv", "ecv"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        row = D.ix[i, 0]
        bn = D.ix[i, 'block_n']
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
            acr = 0
        emv.append(max(Q[m]))
        if Bbins == 101: 
            opb.append(round2(sp.argmax(Q[m]) / 10.0))
        elif Bbins == 11:
            opb.append(round2(sp.argmax(Q[m])))
        if sp.isnan(D.ix[i, 'bid']):
            ecv.append(sp.nan)
            rpe.append(sp.nan)
            coe.append(sp.nan)
            cob.append(sp.nan)
            cmr.append(sp.nan)
            clh.append(sp.nan)
            sve.append(sp.nan)
        else: 
            if Bbins == 101: 
                ind = int(bid * 10) 
                opb.append(round2(sp.argmax(Q[m]) / 10.0))
            elif Bbins == 11:
                ind = int(round(bid))
                opb.append(round2(sp.argmax(Q[m])))
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
            rpei = r - Q[m][ind]
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - Q[m][ind])
            ecv.append(Q[m][ind])
            coe.append(r - 10.0 + cob[-1])
            if project == "econoshyuka" and {'d'} == only_cond:
                ind = int(s1p * 10) 
                p, Q[m] = dr_unilatupd_agent(a[m], b[m], Q[m], ind, 10-ind)
            else: 
                p, Q[m] = dr_unilatupd_agent(a[m], b[m], Q[m], ind, r)
            #p, Q[m] = dr_naive_avf_agent(a[m], b[m], Q[m], ind, r)
            clh.append(sp.log10(p))
            L += sp.log10(p)
            sL[si] += sp.log10(p)
            if Bbins == 11:
                ex11L = sp.log10(0.2) if bid >= 9.5 or bid < 0.5 else sp.log10(0.1)
                L += ex11L
                sL[si] += ex11L
                clh[-1] += ex11L

    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve, emv, ecv)  
        cw.writerows(csvrows)       
        fid.close()

    if len(params) == 2:
        print(a[m], b[m], L, sL)
    elif len(params) == 5:
        print(a[m], b[m], bda, bdb, bdc, L, sL)
    elif len(params) == 9 and 1==0:
        print(a, b, params[6:9], L, sL)
    elif len(params) == 9:
        print(a[m], b[m], bdab, bdc, "({})".format(params[8]), L, sL.values())

    return 10**10 if sp.isnan(L) else -L

def nll_ffx_dr_naive_avf_jac(params, D = D): # deprecated: too complex 
    initQ = {}
    if len(params) == 5:
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bda, bdb, bdc = params[2:5]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 9 and 1==0:
        a = {i[1]: params[0:6:2][i[0]] for i in enumerate(markets)}
        b = {i[1]: params[1:6:2][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[6:9][i[0]]) for i in enumerate(markets)}
    elif len(params) == 9:
        a = {m:params[0] for m in markets}
        b = {m:params[1] for m in markets}
        bdab = params[2:8]
        bdc = params[8]
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ)
    dQ_dx = dict((m, unifB(0)) for m in markets)
    dQ_dc1 = dict((m, beta_distB_da(bda, bdb, bdc))for m in markets)
    dQ_dc2 = dict((m, beta_distB_db(bda, bdb, bdc)) for m in markets)
    dQ_dc3 = dict((m, beta_distB(bda, bdb, 1)) for m in markets)

    jac_a, jac_b, jac_eb1, jac_eb2, jac_eb3 = 5 * (0,)
    for i in range(len(D)):
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
            dQ_da[m] = unifB(0);
        if not sp.isnan(D.ix[i, 'bid']): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            Qc = sp.copy(Q[m]) 
            dQc_da   = sp.copy(dQ_dx) 
            dQc_deb1 = sp.copy(dQ_dc1)   
            dQc_deb2 = sp.copy(dQ_dc2) 
            dQc_deb3 = sp.copy(dQ_dc3) 

            _, Q[m] = dr_naive_avf_agent(a[m], b[m], Q[m], ind, r)
            dQ_da[m][ind] = dQ_da[m][ind] * (1 - Qc[ind]) + r 
            dQ_deb1[m][ind] *= (1 - a[m])  
            dQ_deb2[m][ind] *= (1 - a[m])  
            dQ_deb3[m][ind] *= (1 - a[m])  
 
            jac_a   += b[m]*dQc_da[ind]*(1+r-Qc[ind]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_da[j] for j in B])
            jac_b   += Q[m][ind] - sum([Q[m][j] * p for j in B]) 
            jac_eb1 += b[m]*dQc_deb1[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb1[j] for j in B])  
            jac_eb2 += b[m]*dQc_deb2[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb2[j] for j in B])   
            jac_eb3 += b[m]*dQc_deb3[ind]*(1-a[m]) - b[m] * sum([boltzmann_dist(b[m], Qc, j)*dQc_deb3[j] for j in B])  
    if len(params) == 5:
        print(a[m], b[m], bda, bdb, bdc, (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    elif len(params) == 9 and 1==0:
        print(a, b, params[6:9], (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    elif len(params) == 9:
        print(a[m], b[m], bdab, bdc, (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3))
    
    if any(sp.isnan(i) for i in (-jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3)):
        return 10**10 #sp.inf
    else:
        return -jac_a, -jac_b, -jac_eb1, -jac_eb2, -jac_eb3

def nll_ffx_pu_avf(params, D = D):
    initQ = {}
    if len(params) == 4:
        b = {m:params[0] for m in markets} 
        bda, bdb, bdc = params[1:4]
        initQ = {m: beta_distB(bda, bdb, bdc) for m in markets}
    elif len(params) == 7:
        b = {i[1]: params[0:3][i[0]] for i in enumerate(markets)}
        initQ = {i[1]: unifB(params[4:7][i[0]]) for i in enumerate(markets)}
    elif len(params) == 8:
        b = {m:params[0] for m in markets} 
        bdab = params[1:7]
        use_bdmax = True 
        if not use_bdmax:
            bdc = params[7]
        elif use_bdmax:
            bdmax = params[7]
            #bdc = newtonraphson_secant(5, (bda, bdb)) 
            try:
                bdc = min([brent(0, 10, (bdab[2*i], bdab[2*i+1], bdmax)) for i in range(3)])
            except ValueError:
                return 10**10 
        initQ = {i[1]: beta_distB(bdab[2*i[0]], bdab[2*i[0]+1], bdc) for i in enumerate(markets)}
    Q = copy.deepcopy(initQ) 

    L = 0
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            Q[m] = sp.copy(initQ[m])
        if not sp.isnan(D.ix[i, 'bid']): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            #p, Q[m] = pu_naive_avf_agent(b[m], Q[m], ind, r)
            p, Q[m] = pu_unilatupd_agent(b[m], Q[m], ind, r)
            L += sp.log10(p)
            if Bbins == 11:
                L += sp.log10(0.2) if bid >= 9.5 or bid < 0.5 else sp.log10(0.1) 
    if len(params) == 4: 
        print(b[m], bda, bdb, bdc, L)
    if len(params) == 7: 
        print(b, params[4:7], L)
    if len(params) == 8: 
        print(b[m], bdab, bdc, "({})".format(params[7]), L)

    return 10**10 if sp.isnan(L) else -L


def nll_null():
    return -sp.log10(null_agent()) * m

counter = 0
def nll_ffx_dr_nudger(params, D = D2):  # pana.nll_ffx_dr_nudger(pana.fp[38][2])
    if len(params) == 2:
        fn = 'behvars_drnudger1.csv'
        a, sig = params
        mu0 = copy.deepcopy(initbid)
        q = dict((m, mu0[m]) for m in markets)
    elif len(params) == 3:
        #nu, nd, sig = params
        #initq = copy.deepcopy(initbid)
        #q = dict((m, initq[m]) for m in markets)
        #fn = 'behvars_nvnudger1.csv'
        fn = 'behvars_drnudger2.csv'
        a, siga, sigr = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 4:
        nu, nd, siga, sigr = params
        initq = copy.deepcopy(initbid)
        q = dict((m, initq[m]) for m in markets)
        fn = 'behvars_nvnudger2.csv'
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
        #fn = 'behvars_drnudger3.csv'
        #p1, p2, siga, sigr = params
        #mu0 = copy.deepcopy(initbid)
        #q = copy.deepcopy(mu0)
    elif len(params) == 5 and 1==0:
        fn = 'behvars_drnudger4.csv'
        a, nd, nu, siga, sigr = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 7:
        fn = 'behvars_drnudger5.csv'
        a, nd, nu, siga, sigr, sign, raitc = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 5:
        if project == 'econoshyuka':
            fn = 'behvars_drlepkurnudger7_'+list(only_cond)[0]+'.csv'
        else:
            fn = 'behvars_drlepkurnudger7.csv'
        a, siga, sigr, sign, raitc = params
        mu0 = copy.deepcopy(initbid)
        q = copy.deepcopy(mu0)
    elif len(params) == 5 and 1==0:
        a = params[0] 
        sig = params[1] 
        mu0 = {i[1]: params[2:5][i[0]] for i in enumerate(markets)}
        q = dict((m, mu0[m]) for m in markets)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    opb, rpe, clh, cob, coe, cmr, sve = [], [], [], [], [], [], []
    writevars = True
    if writevars:

        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = mu0[m]
            acr = 0
            prevacc = {m:1 for m in markets}
            prevcob = {m:mu0[m] for m in markets}
        
        opb.append(q[m])
        if sp.isnan(D.ix[i, 'bid']): 
            cob.append(sp.nan)
            rpe.append(sp.nan)
            cmr.append(sp.nan)
            coe.append(sp.nan)
            clh.append(sp.nan)
            sve.append(sp.nan)
        else:  
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
            rpei = r - 10.0 + q[m]
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - 10 + q[m])
            #p, q[m] = naive_gausnudger1_agent(nu, nd, sig, q[m], bid, r) # gaus/lepkur nudger
            #p, q[m] = naive_lepkurnudger2_agent(nu, nd, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0  # gaus/lepkur nudger
            #p, q[m] = dr_gausnudger1_agent(a, sig, q[m], bid, r) # gaus
            #p, q[m] = dr_gausnudger2_agent(a, siga, sigr, q[m], bid, r, prevacc); prevacc[m] = 1 if r > 0 else 0  # gaus
            #p, q[m] = dr_lepkurnudger1_agent(a, sig, q[m], bid, r) # lepkur
            #p, q[m] = dr_lepkurnudger2_agent(a, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 # lepkur
            #nd, nu = p1, p2; p, q[m] = dr_lepkurnudger31_agent(nd, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #a, nu = p1, p2; p, q[m] = dr_lepkurnudger32_agent(a, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #aa, ar = p1, p2; p, q[m] = dr_lepkurnudger33_agent(aa, ar, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger4_agent(a, nd, nu, siga, sigr, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger5_agent(a, nd, nu, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            p, q[m] = dr_lepkurnudger7_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 
            #p, q[m] = dr_fb1_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevcob[m], cob[-1]); prevcob[m] = cob[-1];  
            #p, q[m] = dr_fb2_lepkurnudger6_agent(a, siga, sigr, sign, raitc, q[m], bid, r, prevcob[m], cob[-1]); prevcob[m] = cob[-1]; prevacc[m] = 1 if r>0 else 0  
            clh.append(sp.log(p))
            L += sp.log10(p) 
            sL[si] += sp.log10(p)
 
    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve)  
        cw.writerows(csvrows)       
        fid.close()
    
    global counter 
    counter += 1
    if project == 'neuroshyuka' or project == 'econoshyuka':# and counter % 100 == 0:
        if len(params) == 2:
            print(a, sig, L, sL)
        elif len(params) == 3:
            #print(a, sig, mu0, L)
            print(a, siga, sigr, L, sL)
        elif len(params) == 4:
            #print(p1, p2, siga, sigr, L, sL)
            print(nu, nd, siga, sigr, L, sL)
        elif len(params) == 5:
            print(a, siga, sigr, sign, raitc, L, sL)
        elif len(params) == 5:
            print(a, nd, nu, siga, sigr, L, sL)
        elif len(params) == 5:
            print(a, sig, mu0, L, sL)
        if len(params) == 7: 
            print(a, nd, nu, siga, sigr, sign, raitc, L, sL)
    
    return 10**10 if sp.isnan(L) or sp.isinf(L) else -L


# Kernel density estimator

def nll_ffx_kde(params, D = D):  
    if project == 'econoshyuka':
        fn = 'behvars_kde_'+list(only_cond)[0]+'.csv'
    else:
        fn = 'behvars_kde.csv'

    if len(params) == 2:
      beta, bw = params
    elif len(params) == 1:
      beta = params[0]
      bw = 'scott' # 'silverman'
  
    initbidl = {k:[v] for k, v  in initbid.items()}
    s1est = copy.deepcopy(initbidl); s2est = copy.deepcopy(initbidl); b2est = copy.deepcopy(initbidl)

    if project == "econoshyuka":
        fn = 'econo_' + fn

    opb, rpe, clh, cob, coe, cmr, sve = [], [], [], [], [], [], []
    writevars = True
    if writevars:
        fid = open(fn, 'w') 
        cw = csv.writer(fid, delimiter = ',')
        cw.writerow(["opb", "clh", "rpe", "cob", "coe", "cmr", "sve"])

    L = 0
    sL = {i:0 for i in D.sid.unique()}
    for i in D.index:
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        b2b = round2(D.ix[i, 'b2_bid'])
        s1p = round2(D.ix[i, 's1_rp'])
        s2p = round2(D.ix[i, 's2_rp'])
        r = D.ix[i, 'profit']
        si = D.ix[i, 'sid']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            opbi = initbid[m]
            acr = 0
            s1est = copy.deepcopy(initbidl); s2est = copy.deepcopy(initbidl); b2est = copy.deepcopy(initbidl)
        if sp.isnan(D.ix[i, 'bid']): 
            cob.append(sp.nan); rpe.append(sp.nan); cmr.append(sp.nan); coe.append(sp.nan); clh.append(sp.nan); sve.append(sp.nan)
        else:  
            if m == 'NC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(s1p) 
                else: 
                    if bid >= s1p: 
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
                s2orb2est = None
            elif m == 'SC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(min(s1p, s2p)) 
                else: 
                    if bid >= min(s1p, s2p):
                        cob.append(bid);   
                    else:
                        cob.append(bid + 1); # this is an approximation
                s2orb2est = s2est[m]
            elif m == 'BC':
                if project == "econoshyuka" and {'d'} == only_cond:
                    cob.append(max(s1p, b2b + 0.1)) 
                else: 
                    cob.append(b2b + 0.1);
                s2orb2est = b2est[m]
            rpei = r - 10.0 + opbi
            rpe.append(rpei)
            acr += r
            cmr.append(acr)
            coe.append((10.0 - cob[-1]) - r)
            sve.append((10.0 - cob[-1]) - 10 + opb)

            
            p, opbi, s1est[m], s2orb2est, U = kde_agent(beta, bid, i, m, bw, s1est[m], s2orb2est); 
            if m == 'SC':
                s2est[m] = s2orb2est  
            elif m == 'BC':
                b2est[m] = s2orb2est  

            opb.append(opbi)
            clh.append(sp.log(p))
            L += sp.log10(p) 
            sL[si] += sp.log10(p)
        print(only_cond, beta, bw, i, p, L, sL)
 

        #plt.plot(d.pdf(sp.linspace(0, 10, Bbins)))
        #plt.show()
             
    if writevars:
        csvrows = zip(opb, clh, rpe, cob, coe, cmr, sve)  
        cw.writerows(csvrows)       
        fid.close()
    
    global counter 
    counter += 1
    if project == 'neuroshyuka' or project == 'econoshyuka' and counter % 100 == 0:
        print(only_cond, beta, bw, L, sL)
    
    return 10**10 if sp.isnan(L) or sp.isinf(L) else -L



# Objective function: Sum of squared residuals function / Least squares
def ls_ffx_dr_max(params):
    def dr_max_agent(a, Q, bi, r):
        maxvb = Q.index(max(Q)) 
        Q[bi] += a * (r - Q[bi])
        return maxvb, Q
    def nud_max_agent(a, Q, bi, r):
        maxvb = Q.index(max(Q)) 
        Q[bi] += a * (r - Q[bi])
        return maxvb, Q
    S = 0
    Q = sp.copy(initQ)
    a, b = params[0], params[1]
    for i in range(len(D)):
        r = D.ix[i, 'profit']
        bid = D.ix[i, 'bid']
        if not sp.isnan(bid): 
            if Bbins == 101: 
                ind = int(bid * 10) 
            elif Bbins == 11:
                ind = int(round(bid))
            maxvb, Q = dr_max_agent(a, Q, ind, r)
            s = (bid - maxvb)**2
            S += s 
    return sp.inf if sp.isnan(L) else -L


## Bayesian estimation of density

def emnll():
    """http://nbviewer.ipython.org/github/tritemio/notebooks/blob/master/Mixture_Model_Fitting.ipynb"""
    """ not applicable """

def nll_ffx_kalman_nudger(params):
    if len(params) == 2:
        s0 = {m: params[0] for m in markets}
        mu0 = copy.deepcopy(initbid)
        sb0 = {m: params[1] for m in markets}
        q = dict((m, [mu0[m], sb0[m]]) for m in markets)
    if len(params) == 3:
        s0, mu0, sb0 = params
        q = dict((m, [mu0, sb0]) for m in markets)
        #s0 = {m: params[0] for m in markets}
        #mu0 = copy.deepcopy(initbid)
        #sb0a = {m: params[1] for m in markets}
        #sb0r = {m: params[2] for m in markets}
        #q = dict((m, [mu0[m], sb0a[m], sb0r[m]]) for m in markets)
    elif len(params) == 5:
        s0 = {m: params[0] for m in markets}
        mu0 = {i[1]: params[1:4][i[0]] for i in enumerate(markets)}
        sb0 = {m: params[4] for m in markets}
        q = dict((m, [mu0[m], sb0[m]]) for m in markets)

    L = 0
    for i in range(len(D)):
        m = D.ix[i, 'snb']
        bid = round2(D.ix[i, 'bid'])
        r = D.ix[i, 'profit']
        if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
            q[m] = [mu0, sb0] 
            #q[m] = [mu0[m], sb0a[m], sb0r[m]]; prevacc = {m:1 for m in markets} # mu0, sb0a, sb0r, are updated independenlty for each market
        if not sp.isnan(D.ix[i, 'bid']):  
            p, q[m][0], q[m][1] = kalman_lepkurnudger1_agent(q[m][0], q[m][1], s0, bid, r) # lepkur/gaus
            #p, q[m][0], q[m][1], q[m][2] = kalman_lepkurnudger2_agent(q[m][0], q[m][1], q[m][2], s0[m], bid, r, prevacc[m]); prevacc[m] = 1 if r > 0 else 0 # lepkur/gaus
            L += sp.log10(p) 

    if len(params) == 2:
        print(s0[m], sb0[m], L)
    if len(params) == 3:
        print(s0, mu0, sb0, L)
        #print s0[m], sb0a[m], sb0r[m], L)
    if len(params) == 5:
        print(s0[m], mu0, sb0[m], L)

    return 10**10 if sp.isnan(L) else -L

# Bayesian estimator of static srpd
def bayesian_estimator_ffx(i):
    """https://en.wikipedia.org/wiki/Beta_distribution"""
    initQ = beta_distB(2, 3, 4) 
    a, b = 0.1, 2
    pars0 = sp.concatenate(sp.array([a, b]), initQ, axis = 0)
   
    mms_a = (0, 1.1, 0.25)
    mms_b = (0.1, 2.1, 0.5)
    Ln = llhf_explicit(mms_a, mms_b) 
    for a in Ln.dom_a:
        for b in Ln.dom_b:
            for i in range(len(D)):
                row = D.ix[i, 0]
                m = D.ix[i, 'snb']
                bid = round2(D.ix[i, 'bid'])
                if D.ix[i, 'block_n'] == 1: # restart pars to initial guess
                    Q[m] = sp.copy(initQ)
                r = D.ix[i, 'profit']
                if not sp.isnan(D.ix[i, 'bid']): 
                    if Bbins == 101: 
                        ind = int(bid * 10) 
                    elif Bbins == 11:
                        ind = int(round(bid)) 
                    p, Q[m] = pu_naive_avf_agent(b, Q[m], ind, r)
                    #p, Q[m] = dr_naive_avf_agent(a, b, Q[m], ind, r)
                    Ln.update(a, b, sp.log10(p))
                    print(row, a, b, m, bid, r, Q[m], sp.log10(p))


# Objective functions: Random effects (summary statistics) likelihood functions
""" Suppose that we know the true values of the individual subject parameters α i and β i : for instance, 
suppose we could estimate these perfectly from the choices. In this case, we could estimate the population 
parameters directly from the subject parameters, since Equation 7 reduces to 
P ( μ α , σ α | α 1 . . . α N ) ∝ ∏ i [ P ( α i | μ α , σ α )] · P ( μ α , σ α ) and similarly for β i . 
    Moreover, assuming the distributions P ( α i | μ α , σ α ) and P ( β i | μ β , σ β ) are Gaussian, then 
finding the population parameters for these expressions is just the familiar problem of estimating a Gaussian 
distribution from samples. In particular, the population means and variances can be estimated in the normal 
way by the sample statistics. Importantly, we could then compare the estimated mean parameters between groups
or (within a group) against a constant using standard t-tests. Note that in this case, since the parameter 
estimates arise from an average of samples, confidence intervals can be derived from the sample standard 
deviation divided by the square root of the number of samples, i.e. the familiar standard error of the mean 
in Gaussian estimation. We need not use the Hessian of the underlying likelihood function in this case.
    The procedure would be correct for Gaussian distributed parameters, if the uncertainty about the 
within-subject parameters were negligible. What is the effect of using this as an approximation when this 
uncertainty is instead substantial, as when the parameters were estimated from individual subject model fits? 
Intuitively, the within-subject estimates will be jittered with respect to their true values due to estimation 
noise. We might imagine (and in some circumstances, it is indeed the case) that in computing the population 
means, μ α and μ β , this jitter will average out and the resulting estimates will be unbiased. However, the 
estimation noise in the individual parameter estimates will inflate the estimated population variances beyond 
their true values.
    What mostly matters for our purposes is the validity of t-tests and confidence intervals on the estimated
population means. For some assumptions about the first-level estimation process, Holmes and Friston (1998) 
demonstrate that for t-tests and confidence intervals, the inflation in the population variance is expected to 
be of just the right amount to compensate for the unaccounted uncertainty in the subject-level parameters. 
While this argument is unlikely to hold exactly for the sorts of computational models considered here, it also 
seems that this procedure is relatively insensitive to violations of the assumptions (Friston et al., 2005). 
Thus, these considerations provide at least partial justification for use of the summary statistics procedure.  
"""

def mle_rfx_optmin_batch():
   
    def nll_ffx_1s(params):
        sD = D[D['sid'] == s]
        # nll_ffx_dr_avf  nll_ffx_dr_nudger nll_ffx_kde nll_ffx_dr_naive_nudger
        return nll_ffx_dr_avf(params, sD)

    def mle_rfx_optmin(s):
        init_pars = [0.2, 1] # yoked, dr, beta_dist(3) with mean from initbid
        bounds = [(0, 1), (0.01, 10)] 
        #init_pars = [0.2, 1] # dr, unilatupd
        #bounds = [(0, 1), (0.01, 10)] 
        #init_pars = [1, 1, 1, 1] # naive_lepkurnudger   
        #bounds = [(-10, 10), (-10, 10), (0, 10), (0, 10)] 
        #init_pars = [0.1, 1] # 11, drgausnudger1 
        #bounds = [(0, 1), (0, 10)] 
        #init_pars = [0.1, 1, 1, 1, 0.1] # 111111, dr_nudger7   
        #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 1)] 
        #init_pars = [1, 1] # kde
        #bounds = [(0.1, 10), (0.1, 1)] 
        nllf = nll_ffx_1s
        method = 'L-BFGS-B'
        do_basinhopping = 0
        if not do_basinhopping:
            count = 0 
            def callbackf(params):
                nonlocal count
                print(count)
                if count % 10 == 0:
                    print(params)
                count += 1
            sp.optimize.show_options('minimize', method.lower())
            res = sp.optimize.minimize(nllf, sp.array(init_pars), bounds = bounds, 
                                       method = method, # 'L-BFGS-B' 'TNC'
                                       options = {'disp': True, 'maxiter': 10**8},
                                       callback = callbackf)
            #print(res.hess, res.hess_inv)
        else:
            minimizer_kwargs = {'method': 'L-BFGS-B', 
                                'bounds': bounds, 
                                'options':{'disp': True, 'maxiter': 10**8}}
            res = sp.optimize.basinhopping(nllf, sp.array(init_pars),
                                           minimizer_kwargs = minimizer_kwargs, 
                                           niter = 10, T = 10, stepsize =  0.1,
                                           callback = print)
        return res

    R = dict()
    S = D['sid'].unique()
    for s in S:
        print('\nSubject: ', s, '\n') 
        R[s] = mle_rfx_optmin(s)
    print(R)

    isconv = [v.success for v in R.values()]
    ll_s = [v.fun for v in R.values()]
    par_s = [v.x for v in R.values()]
    ll_m = (sp.mean(ll_s), stats.sem(ll_s, ddof=1))
    par_m = [(sp.mean([p[i] for p in par_s]), stats.sem([p[i] for p in par_s], ddof=1)) for i in range(len(par_s[0]))]
    print(par_m)

    with open('rfx_' + project + '_' + next(iter(only_cond)) + '.pkl', 'wb') as outpkl:
        pickle.dump((R, isconv, ll_m, par_m), outpkl)
    # pickle.load(open(datafile, 'rb'))

    return R


# Objective functions: Mixed effects (full hierarchical model) likelihood functions
""" We can extend our approach of modeling the data generation process explicitly to incorporate a model of 
  how parameters vary across the population (Penny and Friston, 2004).
     Adopting a model of the parameters in the population gives us a two-level hierarchical model of how a 
  full dataset is produced: Each subject’s parameters are drawn from population distributions, then the Q 
  values and the observable choice data are generated, as before, according to an RL model with those 
  parameters.
     The full equation that relates these population-level parameters to a particular subject’s choices, c_i, 
  is then the probability given to them by the RL model, here abbreviated P(c_i | α_i, β_i), averaged over 
  all possible settings of the individual subject’s parameters according to their population distribution:
    P(c_i | μ_α, μ_β, σ_α, σ_β) = dα_i dβ_i P(α_i | μ_α, σ_α) P(β_i | μ_β, σ_β) P(c_i | α_i, β_i)
  This formulation emphasizes that individual parameters α i and β i intervene between the observable quantity 
  and the quantity of interest, but from the perspective of drawing inferences about the population parameters, 
  they are merely nuisance variables to be averaged out. The probability of a full dataset consisting of choice 
  sets c_1 . . . c_N for N subjects is just the product over subjects:
    P ( c 1 . . . c N | μ α , μ β , σ α , σ β ) = ∏_i P(c_i | μ_α, μ_β, σ_α, σ_β)  (6)
     We can then use Bayes’ rule to recover the population parameters in terms of the full dataset:
    P(μ_α, μ_β, σ_α, σ_β | c_1 ... c_N) ∝ P(c_1 ... c_N | μ_α, μ_β, σ_α, σ_β) P(μ_α, μ_β, σ_α, σ_β)  (7)
"""


#################################
###  CHT Level 1:    sellers  ###
#################################
# assumes that sellers try to maximize their profit but disregarding buyers 1st order intentions
# to wit, sellers decide after estimating buyers bid density (bbd)
# this requires the buyers to estimate one's own density (Belief learning)

""" this would affect the initial parameters and their updating only for the Bayesian solution concepts,
    not for the simple Nash equilibria.
"""



################################
###  Numerical optimization  ###
################################

def mle_spsmlhm():
    class MleDelboltz(GenericLikelihoodModel):
        def __init__(self):
            super(MleDelboltz, self).__init__(endog, **kwds)
            #same as GenericLikelihoodModel.__init__(self, endong, exog, **kwds)
        def ll_dr_boltz_gen(bid, r, a, b):
            L = 0
            Q = sp.copy(initQ)
            bid = round2(bid)
            if Bbins == 101:
                i = int(bid * 10) 
            elif Bbins == 11:
                i = int(round(bid))
            p, Q = dr_naive_avf_agent(a, b, Q, i, r)
            L += sp.log10(p)  
            yield L
        def nloglikeobs(self, params):
            a = params[0]
            b = params[1]
            L = self.ll_dr_boltz_gen(self.endog, self.exog, a, b)
            return -L
        def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
            if start_params == None:
                start_params = sp.array([0.1, 0.01])
            return super(MleDelbotz, self).fit(start_params = start_params,
                                               maxiter = maxiter, maxfun = maxfun, **kwds)
    bids = D.ix[:, 'bids']
    gains = D.ix[:, 'profit']
    mod = MleDelbotz(bids, gains)
    res = mod.fit(start_params = (0.1, 0.01))
    return res

def mle_ffx_optmin():
    # Minimize methods
    """http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
       http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

    ### Unconstrained minimization of multivariate scalar functions ###
    Nelder-Mead Simplex algorithm: 'Nelder-Mead'
      Simplest way to minimize a fairly well-behaved function. 
      It requires only function evaluations and is a good choice for simple minimization problems. 
      However, because it does not use any gradient evaluations, it may take longer to find the minimum.
    Powell's method: 'Powell'
      Another optimization algorithm that needs only function calls to find the minimum. 
    Broyden-Fletcher-Goldfarb-Shanno algorithm: 'BFGS'
      This routine uses the gradient of the objective function. If the gradient is not given by the user, 
      then it is estimated using first-differences. The Broyden-Fletcher-Goldfarb-Shanno (BFGS) method 
      typically requires fewer function calls than the simplex algorithm even when the gradient must be estimated. 
    Newton-Conjugate-Gradient algorithm: 'Newton-CG'
      It requires the fewest function calls and is therefore often the fastest method to minimize functions 
      of many variables. This method is a modified Newton's method and uses a conjugate gradient algorithm to 
      (approximately) invert the local Hessian. Newton's method is based on fitting the function locally to a 
      quadratic form.  
      \[ f\left(\mathbf{x}\right)\approx f\left(\mathbf{x}_{0}\right)+\nabla f\left(\mathbf{x}_{0}\right)\cdot\left(\mathbf{x}-\mathbf{x}_{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x}_{0}\right)^{T}\mathbf{H}\left(\mathbf{x}_{0}\right)\left(\mathbf{x}-\mathbf{x}_{0}\right).\]
      To take full advantage of the Newton-CG method, a function which computes the Hessian must be provided. 
      The Hessian matrix itself does not need to be constructed, only a vector which is the product of the Hessian 
      with an arbitrary vector needs to be available to the minimization routine. As a result, the user can provide
      either a function to compute the Hessian matrix, or a function to compute the product of the Hessian with an
      arbitrary vector.

    ### Constrained minimization of multivariate scalar functions ###
    Simulated annealing: 'Anneal'
      It is a probabilistic metaheuristic algorithm for global optimization. 
      It uses no derivative information from the function being optimized.
    Dog-leg trust-region algorithm for unconmin: 'dogleg'
      This algorithm requires the gradient and Hessian; furthermore the Hessian is required to be positive definite.
    Newton conjugate gradient trust-region algorithm for unconmin: 'trust-ncg'
      This algorithm requires the gradient and either the Hessian or 
      a function that computes the product of the Hessian with a given vector.
    Method L-BFGS-B: 'L-BFGS-B'
      For bound constrained minimization.
    Truncated Newton algorithm: 'TNC'
      To minimize a function with variables subject to bounds. 
      This algorithm uses gradient information; it is also called Newton Conjugate-Gradient. 
      It differs from the Newton-CG method described above as it wraps a C implementation and allows each variable 
      to be given upper and lower bounds.
    Constrained Optimization BY Linear Approximation: 'COBYLA'
      The algorithm is based on linear approximations to the objective function and each constraint. 
      The method wraps a FORTRAN implementation of the algorithm.
    Sequential Least Squares Programming Optimization Algorithm: 'SLSQP'
      This algorithm allows to deal with constrained minimization problems of the form:
      \begin{eqnarray*} \min F(x) \\ \text{subject to } & C_j(X) =  0  ,  &j = 1,...,\text{MEQ}\\
         & C_j(x) \geq 0  ,  &j = \text{MEQ}+1,...,M\\
         &  XL  \leq x \leq XU , &I = 1,...,N. \end{eqnarray*} 
       It minimizes a function of several variables with any combination of bounds, equality and inequality constraints.
       The method wraps the SLSQP Optimization subroutine originally implemented by Dieter Kraft.
       Note that the wrapper handles infinite values in bounds by converting them into large floating values.  

    ### Least square fitting: 'leastsq()' ###
      All of the previously-explained minimization procedures can be used to solve a least-squares problem provided 
      the appropriate objective function is constructed. 
      For example, suppose it is desired to fit a set of data {xi,yi} to a known model, y=f(x,p) where p is a vector 
      of parameters for the model that need to be found. A common method for determining which parameter vector gives 
      the best fit to the data is to minimize the sum of squares of the residuals. The residual is usually defined
      for each observed data-point as
        \[ei(p,yi,xi)=∥yi−f(xi,p)∥.\]
      An objective function to pass to any of the previous minization algorithms to obtain a least-squares fit is.
      \[J(p)=∑i=0N−1e2i(p).\]
      The leastsq algorithm performs this squaring and summing of the residuals automatically. It takes as an input
      argument the vector function e(p) and returns the value of p which minimizes J(p)=eTe directly. The user is also
      encouraged to provide the Jacobian matrix of the function (with derivatives down the columns or across the rows).
      If the Jacobian is not provided, it is estimated.
 
    ### Global optimization ###
      http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.basinhopping.html
    """
    init_pars = [0.2, 1] # yoked, dr, beta_dist(3) with mean from initbid
    bounds = [(0, 1), (0.01, 10)] # yoked, dr, beta_dist(3) with mean from initbid
    #init_pars = [0.2, 1, 5, 2, 4] # yoked, dr, beta_dist(3)
    #bounds = [(0, 1), (0.01, 10), (1, 100), (1, 100), (1, 6)] # yoked, dr, beta_dist(3)
    #init_pars = [1, 10, 10, 5] # yoked, pu, beta_dist(3)
    #bounds = [(0.01, 10), (1, 100), (1, 100), (1, 6)] # yoked, pu, beta_dist(3)
    #init_pars = [1, 1, 1, 1] # yoked, naive_gausnudger, initq(1) 
    #bounds = [(-10, 10), (-10, 10), (0, 10), (0, 10)] # yoked, naive_gausnudger, initq(1)
    #init_pars = [1, 5, 1] # 111, kalman,dr_nudger 
    #bounds = [(0, 10), (0, 10), (0, 10)] # 111, kalman,dr_nudger
    #init_pars = [0.2, 2, 5, 2, 5, 2, 5, 2, 5] # 33003, dr, unif(3)
    #bounds = [(0, 1), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (2, 10)] # 33003, dr, unif(3)
    #init_pars = [0.5, 1, 5, 2, 5, 2, 5, 2, 5] # 11331, dr, bd(3)
    #bounds = [(0, 1), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (2, 6)]  # 11331, dr, bd(3)
    #init_pars = [1, 5, 2, 5, 2, 5, 2, 5] # 1331, pu, bd(3)
    #bounds = [(0.01, 10), (0, 10), (1, 8), (0, 10), (1, 8), (0, 10), (1, 8), (2, 10)] # 1331, pu, bd(3)
    #init_pars = [1, 5, 5, 5, 1] # 131, kalman 
    #bounds = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)] # 131, kalman
    #init_pars = [0.5, 1, 5, 5, 5] # 113, dr_nudger1 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 10)] # 113, dr_nudger1
    #init_pars = [0.1, 1, 5, 5, 5] # 113, dr_nudger2 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 10)] # 113, dr_nudger2
    #init_pars = [0.1, 1] # 11, dr_nudger1 
    #bounds = [(0, 1), (0, 10)] # 11, dr_nudger1
    #init_pars = [0.1, 1, 1] # 111, dr_nudger2 
    #bounds = [(0, 1), (0, 10), (0, 10)] # 111, dr_nudger2
    #init_pars = [0.1, 0.1, 1, 1] # 1111, dr_nudger3 
    #bounds = [(0, 1), (-10, 10), (0, 10), (0, 10)] # 1111, dr_nudger3
    #init_pars = [0.1, 0.1, 0.1, 1, 1] # 11111, dr_nudger4 
    #bounds = [(0, 1), (-10, 10), (-10, 10), (0, 10), (0, 10)] # 11111, dr_nudger4
    #init_pars = [0.1, 0.1, 0.1, 1, 1, 1, 0.1] # 11111111, dr_nudger5 
    #bounds = [(0, 1), (-10, 10), (-10, 10), (0, 10), (0, 10), (0, 10), (0, 1)] # 1111111, dr_nudger5
    #init_pars = [0.1, 1, 1, 1, 0.1] # 111111, dr_nudger6 
    #bounds = [(0, 1), (0, 10), (0, 10), (0, 10), (0, 1)] # 11111, dr_nudger6
    #init_pars = [1, 1, 1] # 111, naive_nudger1
    #bounds = [(0.1, 10), (0, 10), (0, 10)] # 111, naive_nudger1
    #init_pars = [1, 1] # 131, kalman1 
    #bounds = [(0, 10), (0, 10)] # 11, kalman1
    #init_pars = [1, 1, 1] # 111, kalman2 
    #bounds = [(0, 10), (0, 10), (0, 10)] # 111, kalman2
    #init_pars = [1, 1, 1, 1] # 1111, naive_nudger2
    #bounds = [(0, 10), (0, 10), (0, 10), (0, 10)] # 1111, naive_nudger2
    #init_pars = [1, 1] # kde
    #bounds = [(0.1, 10), (0.1, 1)] # kde
    #init_pars = [1] # kde
    #bounds = [(0.1, 10)] # kde
    #jac = nll_ffx_dr_naive_avf_jac
    method = 'L-BFGS-B'
    options = {'disp': True, 'maxiter': 10**8}
    nllf = nll_ffx_dr_avf #  nll_ffx_dr_avf nll_ffx_dr_nudger nll_ffx_kde

    do_basinhopping = 1
    if not do_basinhopping:
        sp.optimize.show_options('minimize', method.lower())
        res = sp.optimize.minimize(nllf, sp.array(init_pars), 
                                   method = method, bounds = bounds, 
                                   options = options)
        #print(res.hess, res.hess_inv)
    else:
        minimizer_kwargs = {'method': method, 'bounds': bounds, 'options': options}
        res = sp.optimize.basinhopping(nllf, sp.array(init_pars),
                                       minimizer_kwargs = minimizer_kwargs, 
                                       niter = 10, T = 10, stepsize =  0.1)
    return res


def ssr_optls():
    """http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq"""
    x0 = [0.1, 1]
    func = 0
    bounds = [(0, 1), (10**-3, 10**2)]
    res = sp.optimize.leastsq(func, x0)
    print(res.x, '\n', res.conv_x, '\n', res.mesg, '\n', res.ler)
    return res



from .modselec import *
from .simul_plt import *
