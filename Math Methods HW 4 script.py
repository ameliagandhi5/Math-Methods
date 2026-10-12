#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:41:09 2026

@author: ameliagandhi
"""

# imports 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib as mpl
import scipy 

class ODESolver(object):
    """
    Superclass for numerical methods solving scalar and vector ODEs
      du/dt = f(u, t)
    Attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """
    def __init__(self, f):
        if not callable(f):
            raise TypeError('f is %s, not a function' % type(f))
        # For ODE systems, f will often return a list, but
        # arithmetic operations with f in numerical methods
        # require that f is an array. Let self.f be a function
        # that first calls f(u,t) and then ensures that the
        # result is an array of floats.
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def advance(self):
        """Advance solution one time step."""
        raise NotImplementedError

    def set_initial_condition(self, U0):
        if isinstance(U0, (float,int)):  # scalar ODE
            self.neq = 1
            U0 = float(U0)
        else:                            # system of ODEs
            U0 = np.asarray(U0)          # (assume U0 is sequence)
            self.neq = U0.size
        self.U0 = U0

        # Check that f returns correct length:
        # try:
        #     f0 = self.f(self.U0, 0)
        # except IndexError:
        #     raise IndexError('Index of u out of bounds in f(u,t) func. Legal indices are %s' % (str(range(self.neq))))
        # if f0.size != self.neq:
        #     raise ValueError('f(u,t) returns %d components, while u has %d components' % (f0.size, self.neq))

    def solve(self, time_points, terminate=None):
        """
        Compute solution u for t values in the list/array
        time_points, as long as terminate(u,t,step_no) is False.
        terminate(u,t,step_no) is a user-given function
        returning True or False. By default, a terminate
        function which always returns False is used.
        """
        if terminate is None:
            terminate = lambda u, t, step_no: False

        if isinstance(time_points, (float,int)):
            raise TypeError('solve: time_points is not a sequence')
        self.t = np.asarray(time_points)
        if self.t.size <= 1:
            raise ValueError('ODESolver.solve requires time_points array with at least 2 time points')

        n = self.t.size
        if self.neq == 1:  # scalar ODEs
            self.u = np.zeros(n)
        else:              # systems of ODEs
            self.u = np.zeros((n,self.neq))

        # Assume that self.t[0] corresponds to self.U0
        self.u[0] = self.U0

        # Time loop
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
            if terminate(self.u, self.t, self.k+1):
                break  # terminate loop over k
        return self.u[:k+2], self.t[:k+2]
    
# rk4 stepper (thanks Brad!)
class RungeKutta4(ODESolver):
    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        dt2 = dt/2.0
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + 0.5*K1, t[k] + dt2)
        K3 = dt*f(u[k] + 0.5*K2, t[k] + dt2)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

#%%
# define the 2nd order ODE as 2 first order ODEs 
def ode_system(f, x, w=50):
    f0, f1 = f
    df0 = f1
    df1 = -w**2/x*f0
    return [df0, df1]

# Boundary Conditions
f0 = 0
fN = 1

# Set up the domain
domain = [1e-6,1]
N = 100 # number of gridpoints
x_points = np.linspace(domain[0], domain[1], N)

#%%
# Problems 1 and 2 

# define values for omega to loop through 
w_list = np.linspace(0.01, 10, 100)

# make a colormap for w, used in plotting
norm = mpl.colors.Normalize(vmin=min(w_list), vmax=max(w_list))
cmap = plt.cm.viridis

# make a figure 
fig, ax = plt.subplots(2, 1, figsize=(8, 7))

# make a list for kinetic energy and u 
E_list = [] 
u_list = [] 
f_list = [] 
df_list = [] 

# loop through different w0: 
for w in w_list: 
    
    # figure out which way to shoot with initial guesses 
    df0 = 0
    df = df0 + .5
    f_g1 = [f0,df0]
    f_g2 = [f0,df]
    
    # Perform numerical integration for guess 1
    RK4 = RungeKutta4(lambda f_g1, x: ode_system(f_g1, x, w))
    RK4.set_initial_condition(f_g1)
    f_1, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
    B = f_1[-1,0]-fN # compute Boundary value error (BVE) at guess 1 for solution at x=1
    
    # Perform numerical integration for guess 1
    RK4 = RungeKutta4(lambda f_g2, x: ode_system(f_g2, x, w))
    RK4.set_initial_condition(f_g2)
    f_2, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
    bc = f_2[-1,0]-fN # compute Boundary value error (BVE) at guess 1 for solution at x=1
    
    df = [df0, df] # boundary condition for the proper guesses 
    B = [B,bc] # boundary value errors at our two guesses 
    
    # finally determine which way we should walk 
    dB = B[1]-B[0]
    if (B[0]*dB < 0):
        d = 0.5
    else:
        d = -0.5
    
    # define condition for the bracket 
    doit = True if (B[0]*B[1] > 0.0) else False
    
    # Keep walking until the bracket is found    
    while doit:      
        
        # initial vector solution 
        df0 = df0 + d
        f = [f0,df0] 
                
        # Perform numerical integration
        RK4 = RungeKutta4(lambda f, x: ode_system(f, x, w))
        RK4.set_initial_condition(f)
        f, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
        
        bc = f[-1,0]-fN
        
        df = [df[-1],df0] #boundary condition 
        B = [B[-1],bc]
        doit = True if (B[0]*B[1] > 0.0) else False
    
    # bisect the bracket until we are satisfied (within a certain tolerance)
    epsilon = 1e-6 # desired level of accuracy 
    
    # Keep bisecting until converged = True
    converged = True if (np.abs(df[1]-df[0]) < epsilon) else False
    
    while (not converged):
        df0 = 0.5*(df[1]+df[0])
        f = [f0,df0]
        RK4 = RungeKutta4(lambda f, x: ode_system(f, x, w))
        RK4.set_initial_condition(f)
        f, x = RK4.solve(x_points)
        bc = f[-1,0]-fN

        if (bc*B[0] > 0.0):      # Replace the bracketing guess that produces the same sign as the new guess
            df = [df0,df[1]]
            B = [bc,B[1]]
        else:
            df = [df[0],df0]
            B = [B[0],bc]
    
        converged = True if (np.abs(df[1]-df[0]) < epsilon) else False    # Desired accuracy met?
    
    # append best f to our list for each omega 
    f_list.append(f[:,0])
    df_list.append(f[:,1])
    
    # next things to do 
    # find u via f = Hu where H = alpha*x => u = f/ax
    alpha = 1 # set alpha = 1 for finding u 
    u = np.array(f[:,0])/(alpha*x_points)
    u_list.append(u)
    
    # find the kinetic energy by integrating 
    E = np.trapezoid(.5*u**2, x=x_points)
    E_list.append(E)

    # plot f and the boundary condition 
    ax[0].plot(x_points, f[:,0], color=cmap(norm(w)), markersize=3)
    
# plot E as a function of omega 
ax[1].plot(w_list, E_list, marker='.', c='k')
    
# prettify!
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) # colormap stuff 
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'$\varpi$')
ax[0].plot(1, 1, marker='*', c='k') # plot the boundary condition
ax[0].set_xlabel('Position  $x$')
ax[0].set_ylabel('f')
ax[1].set_xlabel(r'$\varpi$')
ax[1].set_ylabel('Kinetic Energy  $E$')
ax[1].set_yscale('log')

#%%

# plot sample solutions of u and h 
fig, ax = plt.subplots(2, 1, figsize=(7, 5))

for w, u, df in zip(w_list, u_list, df_list):
    ax[0].plot(x_points, u, color=cmap(norm(w)))
    ax[1].plot(x_points, df/w, color=cmap(norm(w)))

# prettify! 
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) # colormap stuff 
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'$\varpi$')
ax[0].set_ylabel('Velocity u(x)')
ax[1].set_ylabel('Height H(x)')
ax[1].set_xlabel('Position x')
ax[1].set_ylabel('Height H(x)')

#%%
# plot analytic and numerical solutions 
fig, ax = plt.subplots(10, 10, figsize=(30, 25), layout='constrained')

# loop through the different omegas
for w, f, a in zip(w_list, f_list, ax.ravel()): 
    
    # find the analytic solution
    bessel = np.sqrt(x) * scipy.special.j1(2* w*np.sqrt(x_points)) / scipy.special.j1(2* w)
    
    # plot the analytic and 
    a.plot(x_points, bessel, linewidth=4, color=cmap(norm(w)), label='Analytic Solution')
    a.plot(x_points, f, linestyle='-.', c='k', label='Numerical Solution')
    
    # prettify
    a.set_xticks([])
    a.set_title(r'$\omega$'+f' = {w:.2f}')
    a.set_xlim(-.1, 1.1)
    a.scatter(0, 0, c='red')
    a.scatter(1, 1, c='red')
    
    if a == ax.ravel()[0]: 
        a.legend(frameon=False)
        
# prettify 
fig.supxlabel('Position x', fontsize=24)
fig.supylabel('f', fontsize=24)

#%%

# PROBLEM 4

# define the 2nd order ODE as 2 first order ODEs 
# change H(x) to be sin(x) 
def ode_system(f, x, w=50):
    f0, f1 = f
    df0 = f1
    df1 = -w**2/(2*np.sin(x))*f0
    return [df0, df1]

# Boundary Conditions
f0 = 0
fN = 1

# Set up the domain
domain = [1e-6,1]
N = 100 # number of gridpoints
x_points = np.linspace(domain[0], domain[1], N)

#%%
# Problems 1 and 2 

# define values for omega to loop through 
w_list = np.linspace(0.01, 10, 100)

# make a colormap for w, used in plotting
norm = mpl.colors.Normalize(vmin=min(w_list), vmax=max(w_list))
cmap = plt.cm.viridis

# make a figure 
fig, ax = plt.subplots(2, 1, figsize=(8, 7))

# make a list for kinetic energy and u 
E_list = [] 
u_list = [] 
f_list = [] 
df_list = [] 

# loop through different w0: 
for w in w_list: 
    
    # figure out which way to shoot with initial guesses 
    df0 = 0
    df = df0 + .5
    f_g1 = [f0,df0]
    f_g2 = [f0,df]
    
    # Perform numerical integration for guess 1
    RK4 = RungeKutta4(lambda f_g1, x: ode_system(f_g1, x, w))
    RK4.set_initial_condition(f_g1)
    f_1, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
    B = f_1[-1,0]-fN # compute Boundary value error (BVE) at guess 1 for solution at x=1
    
    # Perform numerical integration for guess 1
    RK4 = RungeKutta4(lambda f_g2, x: ode_system(f_g2, x, w))
    RK4.set_initial_condition(f_g2)
    f_2, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
    bc = f_2[-1,0]-fN # compute Boundary value error (BVE) at guess 1 for solution at x=1
    
    df = [df0, df] # boundary condition for the proper guesses 
    B = [B,bc] # boundary value errors at our two guesses 
    
    # finally determine which way we should walk 
    dB = B[1]-B[0]
    if (B[0]*dB < 0):
        d = 0.5
    else:
        d = -0.5
    
    # define condition for the bracket 
    doit = True if (B[0]*B[1] > 0.0) else False
    
    # Keep walking until the bracket is found    
    while doit:      
        
        # initial vector solution 
        df0 = df0 + d
        f = [f0,df0] 
                
        # Perform numerical integration
        RK4 = RungeKutta4(lambda f, x: ode_system(f, x, w))
        RK4.set_initial_condition(f)
        f, x = RK4.solve(x_points)    # Integrate!  u is now the solution at all of the grid points.
        
        bc = f[-1,0]-fN
        
        df = [df[-1],df0] #boundary condition 
        B = [B[-1],bc]
        doit = True if (B[0]*B[1] > 0.0) else False
    
    # bisect the bracket until we are satisfied (within a certain tolerance)
    epsilon = 1e-6 # desired level of accuracy 
    
    # Keep bisecting until converged = True
    converged = True if (np.abs(df[1]-df[0]) < epsilon) else False
    
    while (not converged):
        df0 = 0.5*(df[1]+df[0])
        f = [f0,df0]
        RK4 = RungeKutta4(lambda f, x: ode_system(f, x, w))
        RK4.set_initial_condition(f)
        f, x = RK4.solve(x_points)
        bc = f[-1,0]-fN

        if (bc*B[0] > 0.0):      # Replace the bracketing guess that produces the same sign as the new guess
            df = [df0,df[1]]
            B = [bc,B[1]]
        else:
            df = [df[0],df0]
            B = [B[0],bc]
    
        converged = True if (np.abs(df[1]-df[0]) < epsilon) else False    # Desired accuracy met?
    
    # append best f to our list for each omega 
    f_list.append(f[:,0])
    df_list.append(f[:,1])
    
    # next things to do 
    # find u via f = Hu where H = alpha*sin(x) => u = f/asin(x)
    alpha = 1 # set alpha = 1 for finding u 
    u = np.array(f[:,0])/(alpha*2*np.sin(x_points))
    u_list.append(u)
    
    # find the kinetic energy by integrating 
    E = np.trapezoid(.5*u**2, x=x_points)
    E_list.append(E)

    # plot f and the boundary condition 
    ax[0].plot(x_points, f[:,0], color=cmap(norm(w)), markersize=3)
    
# plot E as a function of omega 
ax[1].plot(w_list, E_list, marker='.', c='k')
    
# prettify!
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) # colormap stuff 
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'$\varpi$')
ax[0].plot(1, 1, marker='*', c='k') # plot the boundary condition
ax[0].set_xlabel('Position  $x$')
ax[0].set_ylabel('f')
ax[1].set_xlabel(r'$\varpi$')
ax[1].set_ylabel('Kinetic Energy  $E$')
ax[1].set_yscale('log')

#%%

# plot sample solutions of u and h 
fig, ax = plt.subplots(2, 1, figsize=(7, 5))

for w, u, df in zip(w_list, u_list, df_list):
    ax[0].plot(x_points, u, color=cmap(norm(w)))
    ax[1].plot(x_points, df/w, color=cmap(norm(w)))

# prettify! 
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) # colormap stuff 
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r'$\varpi$')
ax[0].set_ylabel('Velocity u(x)')
ax[1].set_ylabel('Height H(x)')
ax[1].set_xlabel('Position x')
ax[1].set_ylabel('Height H(x)')