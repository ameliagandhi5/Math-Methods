#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:28:13 2026

@author: ameliagandhi
"""

# imports 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib as mpl
import scipy 
from matplotlib.patches import Rectangle

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
# define lorentz system 
def lorentz_system(f_init, t, r, sigma, b):
    '''
    system of ODEs for Lorentz system
 
    Parameters
    ----------
    f_init : list or array of initial x, y, z values for the lorentz system
    x : array of times 
    sigma : parameter (float)
    r : parameter (float)
    b : parameter (float)

    Returns
    -------
    list of derivatives (call this dfdt) 
    '''
    
    x, y, z = f_init
    dxdt = sigma*(y-x)
    dydt = r*x - y - x*z
    dzdt = x*y - b*z

    return [dxdt, dydt, dzdt]

# Set up the domain
t = np.arange(0, 100, .001) # this results in 100 gridpoints

#%%
# PARAMETER SET 1

# define initial conditions at t=0 
f_init = [0, 1, 0]

# run the integration with RK4! 
RK4 = RungeKutta4(lambda f_init, t: lorentz_system(f_init, t, r=17, sigma=3, b=1))
RK4.set_initial_condition(f_init)
f_1, t = RK4.solve(t)

# plot some pretty things!
fig, ax = plt.subplots(3, 1, figsize=(7,10))
ax = ax.ravel()

# plot how x, y, z vary with t 
ax[0].plot(t, f_1[:,0], c='olive', label='x')
ax[0].plot(t, f_1[:,1], c='olivedrab', label='y')
ax[0].plot(t, f_1[:,2], c='yellowgreen', label='z')
ax[0].axhline(-4, c='k', linestyle='--', label='-4')
ax[0].axhline(16, c='gray', linestyle='--', label='16')

# plot y vs z and y vs x 
ax[1].plot(f_1[:,1], f_1[:,2], c='k', label='x')
ax[2].plot(f_1[:,1], f_1[:,0], c='k', label='x')
ax[1].scatter(-4, 16, c='green', marker='x', zorder=10)
ax[2].scatter(-4, -4, c='green', marker='x', zorder=10)

# prettify! 
ax[0].set_title('Lorentz System with Parameter Set 1', fontsize=16)
ax[0].set_ylabel('x, y, z')
ax[0].set_xlabel('t')
ax[0].legend(frameon=False, ncol=5) 
ax[1].set_ylabel('z')
ax[1].set_xlabel('y')
ax[2].set_ylabel('x')
ax[2].set_xlabel('y')

# 3d plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot3D(f_1[:,0], f_1[:,1], f_1[:,2], c='k')
ax.scatter3D(-4, -4, 16, c='green', marker='x')

# prettify 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parameter Set 1')


#%%
# PLAY WITH PARAMETERS 

# define initial conditions at t=0 
f_init = [-100, -100, 100]

# run the integration with RK4! 
RK4 = RungeKutta4(lambda f_init, t: lorentz_system(f_init, t, r=17, sigma=3, b=1))
RK4.set_initial_condition(f_init)
f_1, t = RK4.solve(t)

# plot some pretty things!
fig, ax = plt.subplots(3, 1, figsize=(7,10))
ax = ax.ravel()

# plot how x, y, z vary with t 
ax[0].plot(t, f_1[:,0], c='olive', label='x')
ax[0].plot(t, f_1[:,1], c='olivedrab', label='y')
ax[0].plot(t, f_1[:,2], c='yellowgreen', label='z')
ax[0].axhline(-4, c='k', linestyle='--', label='-4')
ax[0].axhline(16, c='gray', linestyle='--', label='16')

# plot y vs z and y vs x 
ax[1].plot(f_1[:,1], f_1[:,2], c='k', label='x')
ax[2].plot(f_1[:,1], f_1[:,0], c='k', label='x')
ax[1].scatter(-4, 16, c='green', marker='x', zorder=10)
ax[2].scatter(-4, -4, c='green', marker='x', zorder=10)

# prettify! 
ax[0].set_title('Lorentz System Playing with Parameters', fontsize=16)
ax[0].set_ylabel('x, y, z')
ax[0].set_xlabel('t')
ax[0].legend(frameon=False, ncol=5) 
ax[1].set_ylabel('z')
ax[1].set_xlabel('y')
ax[2].set_ylabel('x')
ax[2].set_xlabel('y')

# what I learned manipulating variables 
# 0 < x < 13
# 0 < y < 16
# -9 < z < 6
# breaks when at the origin (stays at the origin)

# 3d plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot3D(f_1[:,0], f_1[:,1], f_1[:,2], c='k')
ax.scatter3D(-4, -4, 16, c='green', marker='x')

# prettify 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parameter Set 1\n with Initial conditions x=-100, y=-100, z=100')

#%%
# PARAMETER SET 2 

# define initial conditions at t=0 
f_init = [0, 1, 0]

# run the integration with RK4! 
RK4 = RungeKutta4(lambda f_init, t: lorentz_system(f_init, t, r=26, sigma=3, b=1))
RK4.set_initial_condition(f_init)
f_1, t = RK4.solve(t)

# plot some pretty things!
fig, ax = plt.subplots(3, 1, figsize=(7,10))
ax = ax.ravel()

# plot how x, y, z vary with t 
ax[0].plot(t, f_1[:,0], c='olive', label='x')
ax[0].plot(t, f_1[:,1], c='olivedrab', label='y')
ax[0].plot(t, f_1[:,2], c='yellowgreen', label='z')
ax[0].axhline(-5, c='k', linestyle='--', label='-5')
ax[0].axhline(5, c='k', linestyle='--', label='5')
ax[0].axhline(25, c='gray', linestyle='--', label='25')

# plot y vs z and y vs x 
ax[1].plot(f_1[:,1], f_1[:,2], c='k', label='x')
ax[2].plot(f_1[:,1], f_1[:,0], c='k', label='x')

# prettify! 
ax[0].set_title('Lorentz System with Parameter Set 2', fontsize=16)
ax[0].set_ylabel('x, y, z')
ax[0].set_xlabel('t')
ax[0].legend(frameon=False, ncol=6) 
ax[1].set_ylabel('z')
ax[1].set_xlabel('y')
ax[2].set_ylabel('x')
ax[2].set_xlabel('y')

# CHARACTERIZE BEHAVIOR OF THE REGION 

# 3d plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
ax.view_init(elev=45, azim=70, roll=0)

ax.plot3D(f_1[:,0], f_1[:,1], f_1[:,2], c='k')
ax.scatter3D(-5, -5, 25, c='green', marker='x')
ax.scatter3D(5, 5, 25, c='green', marker='x')

# prettify 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parameter Set 2')

#%%
# PARAMETER SET 3 

# define initial conditions at t=0 
f_init = [0, 1, 0]

# run the integration with RK4! 
RK4 = RungeKutta4(lambda f_init, t: lorentz_system(f_init, t, r=99.65, sigma=3, b=1))
RK4.set_initial_condition(f_init)
f_1, t = RK4.solve(t)

# plot some pretty things!
fig, ax = plt.subplots(4, 1, figsize=(7,14))
ax = ax.ravel()

z_plane = 100 # this is the plane we will use for the Poincare diagram 

# plot how x, y, z vary with t 
ax[0].plot(t[1000:], f_1[1000:,0], c='olive', label='x')
ax[0].plot(t[1000:], f_1[1000:,1], c='olivedrab', label='y')
ax[0].plot(t[1000:], f_1[1000:,2], c='yellowgreen', label='z')
ax[0].axhline(-5, c='k', linestyle='--', label='-5')
ax[0].axhline(5, c='k', linestyle='--', label='5')
ax[0].axhline(25, c='gray', linestyle='--', label='25')

# plot y vs z and y vs x 
ax[1].plot(f_1[1000:,1], f_1[1000:,2], c='k', label='x', linewidth=.5)
ax[2].plot(f_1[1000:,1], f_1[1000:,0], c='k', label='x', linewidth=.5)
ax[3].plot(f_1[1000:,0], f_1[1000:,2], c='k', label='x', linewidth=.5)
ax[1].axhline(z_plane, c='green', linestyle='--')
ax[3].axhline(z_plane, c='green', linestyle='--')

# prettify! 
ax[0].set_title('Lorentz System with Parameter Set 3', fontsize=16)
ax[0].set_ylabel('x, y, z')
ax[0].set_xlabel('t')
ax[0].legend(frameon=False, ncol=6) 
ax[1].set_ylabel('z')
ax[1].set_xlabel('y')
ax[2].set_ylabel('x')
ax[2].set_xlabel('y')
ax[3].set_ylabel('z')
ax[3].set_xlabel('x')

# CHARACTERIZE BEHAVIOR OF THE REGION 

# 3d plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
ax.view_init(elev=10)

ax.plot3D(f_1[1000:,0], f_1[1000:,1], f_1[1000:,2], c='k', linewidth=.5)
# ax.scatter3D(-5, -5, 25, c='green', marker='x')
# ax.scatter3D(5, 5, 25, c='green', marker='x')

# prettify 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parameter Set 3')

# plot plane used for the Poincare diagram 
X = np.linspace(np.min(f_1[1000:,0]), np.max(f_1[1000:,0]), 50)
Y = np.linspace(np.min(f_1[1000:,1]), np.max(f_1[1000:,1]), 50)
X, Y = np.meshgrid(X, Y)

Z = 0*X + 0*Y + z_plane

ax.plot_surface(X, Y, Z, color='green', alpha=.3)

# plot the points used for the Poincare diagram 
x = f_1[1000:,0]
y = f_1[1000:,1]
z = f_1[1000:,2]
t_poincare = t[1000:]
crossings = np.where((z[:-1] < z_plane) & (z[1:] >= z_plane))[0] # going from below to above z=z_plane

# Interpolate crossing coordinates
f = (z_plane - z[crossings]) / (z[crossings + 1] - z[crossings])
x_cross = x[crossings] + f * (x[crossings + 1] - x[crossings])
y_cross = y[crossings] + f * (y[crossings + 1] - y[crossings])
z_cross = np.full_like(x_cross, z_plane)
t_crossings = t_poincare[crossings]

# plot the points on the 3d plot 
ax.scatter3D(x_cross, y_cross, z_cross, c='green', zorder=10, marker='o')

# plot the points on a Poincare diagram 
fig, ax = plt.subplots(3, 1, figsize=(8, 10)) 
fig.subplots_adjust(hspace=0.5)

sc = ax[0].scatter(x_cross, y_cross, c=t_crossings, s=8, cmap='YlGn', marker='*')
ax[1].scatter(x_cross, y_cross, c=t_crossings, s=8, cmap='YlGn', marker='*')
ax[2].scatter(x_cross, y_cross, c=t_crossings, s=8, cmap='YlGn', marker='*') # 

# prettify the Poincare diagram 
ax[0].set_xlabel('x') 
ax[0].set_ylabel('y') 
ax[1].set_xlabel('x') 
ax[1].set_ylabel('y') 
ax[2].set_xlabel('x') 
ax[2].set_ylabel('y') 

ax[0].set_title('Poincare Diagram', fontsize=16) 
ax[1].set_title('Zoomed In')
ax[2].set_title('Zoomed In More')

# zoom in on the Poincare diagram 
ax[1].set_ylim(-44, -41)
ax[1].set_xlim(-12.2, -11.75)

ax[2].set_ylim(-41.9, -41.7)
ax[2].set_xlim(-11.825, -11.8)

cbar = plt.colorbar(sc, ax=ax, pad=.1)
cbar.set_label('Time')

ax[0].legend(frameon=False)
ax[0].set_facecolor('lightgray')
ax[1].set_facecolor('lightgray')
ax[2].set_facecolor('lightgray')

#%%
# PARAMETER SET 4 
# choose a chaotic regime
# almost periodic behavior with chaotic bursts 

# define initial conditions at t=0 
f_init = [0, 1, 0]

# run the integration with RK4! 
RK4 = RungeKutta4(lambda f_init, t: lorentz_system(f_init, t, r=130.5, sigma=10, b=8/3)) # 
RK4.set_initial_condition(f_init)
f_1, t = RK4.solve(t)

# plot some pretty things!
fig, ax = plt.subplots(4, 1, figsize=(7,14))
ax = ax.ravel()

z_plane = 140 # this is the plane we will use for the Poincare diagram 

# plot how x, y, z vary with t 
ax[0].plot(t[1000:], f_1[1000:,0], c='olive', label='x')
ax[0].plot(t[1000:], f_1[1000:,1], c='olivedrab', label='y')
ax[0].plot(t[1000:], f_1[1000:,2], c='yellowgreen', label='z')
# ax[0].axhline(-5, c='k', linestyle='--', label='-5')
# ax[0].axhline(5, c='k', linestyle='--', label='5')
# ax[0].axhline(25, c='gray', linestyle='--', label='25')

# plot y vs z and y vs x 
ax[1].plot(f_1[1000:,1], f_1[1000:,2], c='k', label='x', linewidth=.5)
ax[2].plot(f_1[1000:,1], f_1[1000:,0], c='k', label='x', linewidth=.5)
ax[3].plot(f_1[1000:,0], f_1[1000:,2], c='k', label='x', linewidth=.5)
ax[1].axhline(z_plane, c='green', linestyle='--')
ax[3].axhline(z_plane, c='green', linestyle='--')

# prettify! 
ax[0].set_title('Lorentz System with Parameter Set 3', fontsize=16)
ax[0].set_ylabel('x, y, z')
ax[0].set_xlabel('t')
ax[0].legend(frameon=False, ncol=6) 
ax[1].set_ylabel('z')
ax[1].set_xlabel('y')
ax[2].set_ylabel('x')
ax[2].set_xlabel('y')
ax[3].set_ylabel('z')
ax[3].set_xlabel('x')

# CHARACTERIZE BEHAVIOR OF THE REGION 

# 3d plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
ax.view_init(elev=10)

ax.plot3D(f_1[1000:,0], f_1[1000:,1], f_1[1000:,2], c='k', linewidth=.5)
# ax.scatter3D(-5, -5, 25, c='green', marker='x')
# ax.scatter3D(5, 5, 25, c='green', marker='x')

# prettify 
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Parameter Set 4')

# plot plane used for the Poincare diagram 
X = np.linspace(np.min(f_1[1000:,0]), np.max(f_1[1000:,0]), 50)
Y = np.linspace(np.min(f_1[1000:,1]), np.max(f_1[1000:,1]), 50)
X, Y = np.meshgrid(X, Y)

Z = 0*X + 0*Y + z_plane

ax.plot_surface(X, Y, Z, color='green', alpha=.3)

# plot the points used for the Poincare diagram 
x = f_1[1000:,0]
y = f_1[1000:,1]
z = f_1[1000:,2]
t_poincare = t[1000:]
crossings = np.where((z[:-1] < z_plane) & (z[1:] >= z_plane))[0] # going from below to above z=z_plane

# Interpolate crossing coordinates
f = (z_plane - z[crossings]) / (z[crossings + 1] - z[crossings])
x_cross = x[crossings] + f * (x[crossings + 1] - x[crossings])
y_cross = y[crossings] + f * (y[crossings + 1] - y[crossings])
z_cross = np.full_like(x_cross, z_plane)
t_crossings = t_poincare[crossings]

# plot the points on the 3d plot 
ax.scatter3D(x_cross, y_cross, z_cross, c='green', zorder=10, marker='o')

# plot the points on a Poincare diagram 
fig, ax = plt.subplots(2, 1, figsize=(8, 6)) 
fig.subplots_adjust(hspace=0.5)

sc = ax[0].scatter(x_cross, y_cross, c=t_crossings, s=8, cmap='YlGn', marker='*')
ax[1].scatter(x_cross, y_cross, c=t_crossings, s=8, cmap='YlGn', marker='*')

# prettify the Poincare diagram 
ax[0].set_xlabel('x') 
ax[0].set_ylabel('y') 
ax[1].set_xlabel('x') 
ax[1].set_ylabel('y') 

ax[0].set_title('Poincare Diagram', fontsize=16) 
ax[1].set_title('Zoomed In')

# zoom in on the Poincare diagram 
ax[1].set_xlim(10, 40)
ax[1].set_ylim(30, 90)

cbar = plt.colorbar(sc, ax=ax, pad=.1)
cbar.set_label('Time')

ax[0].legend(frameon=False)
ax[0].set_facecolor('lightgray')
ax[1].set_facecolor('lightgray')
