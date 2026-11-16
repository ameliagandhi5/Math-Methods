#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:44:27 2026

@author: ameliagandhi
"""

import numpy as np 
from astropy.io import fits 
import matplotlib.pyplot as plt 
import scipy 
import matplotlib.colors as colors
import matplotlib.animation as ani
from IPython.display import Video

#%%

# 1 
# --------------------

# import the fits file 
file_name = '/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Math Methods/hmi.rdVtrack_fd15.2121.225.225.0.-22.5.-00.0.V.fits'
hdu = fits.open(file_name) 
c_sun = 4.37e9
header = hdu['PRIMARY'].header
mapscale = header['MAPSCALE']*c_sun/360 # in meters
cadence = header['CADENCE']
data = hdu['PRIMARY'].data

### make a big figure to plot some images 
fig, ax = plt.subplots(2, 3, figsize=(8, 6), layout='compressed')

# mark indices of images we are interested in 
indices = np.arange(0, 2304, 400)

for a, i in zip(ax.ravel(), indices): 
    pcm = a.imshow(data[i], origin='lower', cmap='hot', extent=[0, np.shape(data)[1]*mapscale, 0, np.shape(data)[1]*mapscale])
    a.set_title(f'Time: {i*cadence:.0f} s')
    a.label_outer()
    
# plot the columns/rows that are used to look at variatins in x and y with 
ax[0, 0].axvline(180*mapscale, c='green')
ax[0, 0].axhline(180*mapscale, c='blue')
    
# prettify 
fig.colorbar(pcm, ax=ax[-1, :], orientation='horizontal', label='Doppler Velocities [m/s]')
ax[1, 1].set_xlabel('x [m]')
ax[0, 0].set_ylabel('y [m]')
ax[1, 0].set_ylabel('y [m]')

### plot how the doppler velocity changes with x and y 

# use row 180 for x and y in the very first image 
img0 = data[0]
img0_180x = img0[180] # this is looking at 180x, varies with y! 
img0_180y = img0[:, 180] # this is looking at 180y, varies with x! 

# plot how doppler velocities vary with x and y 
fig, ax = plt.subplots(2, 1, sharey=True, figsize=(10, 7))
ax[0].plot(np.arange(np.shape(data)[1])*mapscale, img0_180x, label=f'x={180*mapscale:.2e} m', c='green')
ax[1].plot(np.arange(np.shape(data)[1])*mapscale, img0_180y, label=f'y={180*mapscale:.2e} m', c='blue')

### calculate the sound speed of the sun and plot as a horizontal line 
T = 6000 # K 
c_s = np.sqrt(1.67*15680*T) # looked up the values of gamma and the specific gas constant 

print(f'Sound speed {c_s/1e3:.2f} km/s')


### find the root mean square of the velocities 

flattened_data = data.flatten() 
rms = np.sqrt(1/len(flattened_data)*np.sum(flattened_data**2))

print(f'RMS velocity: {rms/1e3:.2f} km/s')

# prettify 
ax[0].legend(frameon=False) 
ax[1].legend(frameon=False) 
ax[0].set_xlabel('y [m]')
ax[1].set_xlabel('x [m]')
ax[0].set_ylabel('Doppler Velocity [m/s]')
ax[1].set_ylabel('Doppler Velocity [m/s]')

#%%

# 2
# --------------------
### compute the fft 
fft_windowless = scipy.fft.fftn(data, workers = -1)
power_windowless = scipy.fft.fftshift(fft_windowless.real**2 + fft_windowless.imag**2)

# window the data 
Nz, Ny, Nx = data.shape  # get the shape of the data 

w_z = scipy.signal.windows.hann (Nz, sym=True)[:, None, None]  # shape: (Nz, 1, 1)
w_y = scipy.signal.windows.hann (Ny, sym=True)[None, :, None]  # shape: (1, Ny, 1)
w_x = scipy.signal.windows.hann (Nx, sym=True)[None, None, :]  # shape: (1, 1, Nx)

data_windowed = w_x * w_y * w_z * data # windowed data 

### compute the fft of the windowed data 
fft_windowed = scipy.fft.fftn(data_windowed, workers = -1)
power_windowed = scipy.fft.fftshift(fft_windowed.real**2 + fft_windowed.imag**2)

#%%
# look at the windowed data
fig, ax = plt.subplots(1, 1)
sc = ax.imshow(data_windowed[Nz//2, :, :], cmap='hot', origin='lower', extent=[0, np.shape(data)[1]*mapscale, 0, np.shape(data)[1]*mapscale])
fig.colorbar(sc, label='Doppler Velocities [m/s]')
ax.set_title('3D Hann Window\nSlice Through Median Time')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

#%%
# get the list of frequencies for the data 
# the f is really omega_t, kx, and ky 
f_t = scipy.fft.fftshift(scipy.fft.fftfreq(Nz, d=cadence)*1000)*2*np.pi # mHz
fy = scipy.fft.fftshift(scipy.fft.fftfreq(Ny, d=mapscale))*2*np.pi
fx = scipy.fft.fftshift(scipy.fft.fftfreq(Nx, d=mapscale))*2*np.pi

# f_t = scipy.fft.fftshift(scipy.fft.fftfreq(Nz)) # mHz
# fy = scipy.fft.fftshift(scipy.fft.fftfreq(Ny))
# fx = scipy.fft.fftshift(scipy.fft.fftfreq(Nx))

#%%

### plot slices of the data 

# plot a slice in omega 
fig, ax = plt.subplots(2, 3, figsize=(10, 6), layout='compressed')

sc = ax[0, 0].imshow(np.log(power_windowless[1650, :, :]), extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)], origin='lower', aspect='auto')

# plot a slice in kx 
ax[0, 1].imshow(np.log(power_windowless[:, 100, :]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(f_t), np.max(f_t)], aspect='auto')

# plot a slice in ky
ax[0, 2].imshow(np.log(power_windowless[:, :, 100]), origin='lower', extent=[np.min(fy), np.max(fy), np.min(f_t), np.max(f_t)], aspect='auto')

### plot windowed data 
# plot a slice in omega
sc = ax[1, 0].imshow(np.log(power_windowed[1650, :, :]), extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)], origin='lower', aspect='auto')

# plot a slice in kx 
ax[1, 1].imshow(np.log(power_windowed[:, 100, :]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(f_t), np.max(f_t)], aspect='auto')

# plot a slice in ky
ax[1, 2].imshow(np.log(power_windowed[:, :, 100]), origin='lower', extent=[np.min(fy), np.max(fy), np.min(f_t), np.max(f_t)], aspect='auto')

# prettify
ax[1, 0].set_xlabel(r'$k_x$ (1/m)')
ax[0, 0].set_ylabel('Raw Data\n' + r'$k_y$ (1/m)')
ax[1, 0].set_ylabel('Windowed Data\n' +r'$k_y$ (1/m)')
ax[1, 1].set_xlabel(r'$k_y$ (1/m)')
ax[0, 1].set_ylabel(r'$\omega$ (mHz)')
ax[1, 1].set_ylabel(r'$\omega$ (mHz)')
ax[1, 2].set_xlabel(r'$k_x$ (1/m)')
ax[0, 2].set_ylabel(r'$\omega$ (mHz)')
ax[1, 2].set_ylabel(r'$\omega$ (mHz)')
ax[0, 0].set_title(rf'$\omega$ = {f_t[192]:.4f} mHz')
ax[0, 1].set_title(rf'$k_x$ = {fx[100]:.4e} 1/m')
ax[0, 2].set_title(rf'$k_y$ = {fy[100]:.4e} 1/m')
fig.colorbar(sc, ax=ax[:,2], label='log(power)') 

#%%
### plot slices of the data with one of the wavenumbers equal to 0 

# plot a slice in omega 
fig, ax = plt.subplots(2, 3, figsize=(10, 6), layout='compressed')

sc = ax[0, 0].imshow(np.log(power_windowless[Nz//2, :, :]), extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)], origin='lower', aspect='auto')

# plot a slice in kx 
ax[0, 1].imshow(np.log(power_windowless[:, 192, :]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(f_t), np.max(f_t)], aspect='auto')

# plot a slice in ky
ax[0, 2].imshow(np.log(power_windowless[:, :, 192]), origin='lower', extent=[np.min(fy), np.max(fy), np.min(f_t), np.max(f_t)], aspect='auto')

### plot windowed data 
# plot a slice in omega
sc = ax[1, 0].imshow(np.log(power_windowed[Nz//2, :, :]), extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)], origin='lower', aspect='auto')

# plot a slice in kx 
ax[1, 1].imshow(np.log(power_windowed[:, 192, :]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(f_t), np.max(f_t)], aspect='auto')

# plot a slice in ky
ax[1, 2].imshow(np.log(power_windowed[:, :, 192]), origin='lower', extent=[np.min(fy), np.max(fy), np.min(f_t), np.max(f_t)], aspect='auto')

# prettify
ax[1, 0].set_xlabel(r'$k_x$ (1/m)')
ax[0, 0].set_ylabel('Raw Data\n' + r'$k_y$ (1/m)')
ax[1, 0].set_ylabel('Windowed Data\n' +r'$k_y$ (1/m)')
ax[1, 1].set_xlabel(r'$k_y$ (1/m)')
ax[0, 1].set_ylabel(r'$\omega$ (mHz)')
ax[1, 1].set_ylabel(r'$\omega$ (mHz)')
ax[1, 2].set_xlabel(r'$k_x$ (1/m)')
ax[0, 2].set_ylabel(r'$\omega$ (mHz)')
ax[1, 2].set_ylabel(r'$\omega$ (mHz)')
ax[0, 0].set_title(rf'$\omega$ = {f_t[Nz//2]:.0f} mHz')
ax[0, 1].set_title(rf'$k_x$ = {fx[192]:.0f} 1/m')
ax[0, 2].set_title(rf'$k_y$ = {fy[192]:.0f} 1/m')
fig.colorbar(sc, ax=ax[:,2], label='log(power)') 

#%%
### make an animation going throuhg omega with the raw data

# create a movie writer object
FFMpegWriter = ani.writers['ffmpeg']
writer = FFMpegWriter(fps=50)

# create a figure, hocus pocus with figure, dpi 
figure, ax = plt.subplots(1, 1)
figure.patch.set_alpha(1.0)
dpi = 150 

# write frames in movie file 
with writer.saving(figure, f"/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Math Methods/fft_movie_raw.mp4", dpi): 
    for i in range(len(f_t)): 
        
        if i % 100 == 0: 
            print(i)
        
        plt.cla()
        sc = ax.imshow(np.log(power_windowless[i]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)])
        
        # prettify 
        if i == 0: 
            fig.colorbar(sc, ax=ax, label='log(power)') 
        ax.set_title(rf'FFT of the Raw Data Sliced at $\omega$ = {f_t[i]:.2f} mHz')
        ax.set_xlabel(r'$k_x$ (1/m)')
        ax.set_ylabel(r'$k_y$ (1/m)')
        
        # write that frame out to the movie file
        writer.grab_frame()
        
#%%
### make an animation going throuhg omega with the windowed data 

# create a movie writer object
FFMpegWriter = ani.writers['ffmpeg']
writer = FFMpegWriter(fps=50)

# create a figure, hocus pocus with figure, dpi 
figure, ax = plt.subplots(1, 1)
figure.patch.set_alpha(1.0)
dpi = 150 

# write frames in movie file 
with writer.saving(figure, f"/Users/ameliagandhi/Library/CloudStorage/OneDrive-Personal/Documents/Grad School/Classes/Math Methods/fft_movie_windowed.mp4", dpi): 
    for i in range(len(f_t)): 
        
        if i % 100 == 0: 
            print(i)
        
        plt.cla()
        sc = ax.imshow(np.log(power_windowed[i]), origin='lower', extent=[np.min(fx), np.max(fx), np.min(fy), np.max(fy)])
        
        # prettify 
        if i == 0: 
            fig.colorbar(sc, ax=ax, label='log(power)') 
        ax.set_title(rf'FFT of the Windowed Data Sliced at $\omega$ = {f_t[i]:.2f} mHz')
        ax.set_xlabel(r'$k_x$ (1/m)')
        ax.set_ylabel(r'$k_y$ (1/m)')
        
        # write that frame out to the movie file
        writer.grab_frame()

#%%

### plot different k's with delta k over omega 

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# pick different ranges of kn via kx 
kn = np.linspace(np.diff(fx)[0]*10, np.average(np.abs(fx)), 5) # lets pick 5 values evenly spaced apart 

# define deltak 
dk = np.diff(fx)[0]*10 # 10 times the frequency bins? 

# calculate what kn is from kx bins 
# kn = np.sqrt(2*kx)
# dk = np.sqrt(2*dkx)

print(f'{dk:.2e}')

print(kn)
 
# loop through the indices where kx is between the bins 
for i in range(len(kn)): 
    
    print(kn[i])
    # find the indices between the bins I defined 
    if i == len(kn)-1:
        condition = (np.abs(fx) >= kn[i] - dk)
    else: 
        condition = (np.abs(fx) >= kn[i] - dk) & (np.abs(fx) <= kn[i] + dk)
    
    # loop through omega 
    power_kn_windowless = [] 
    power_kn_windowed = [] 
    for w in range(len(power_windowless)): 
        
        # get the average power over the annulus we are looking at 
        power_windowless_k_annulus = np.log(np.average(power_windowless[w, condition, condition]))
        
        # do the same thing for the windowed power 
        power_windowed_k_annulus = np.log(np.average(power_windowed[w, condition, condition])) 
    
        # take an average of this and store in an array 
        power_kn_windowless.append(power_windowless_k_annulus)
        power_kn_windowed.append(power_windowed_k_annulus)
    
    # plot omega vs power at each k value 
    ax[0].plot(f_t, power_kn_windowless, label=rf'$k_n$: {kn[i]:.2e}')
    ax[1].plot(f_t, power_kn_windowed, label=rf'$k_n$: {kn[i]:.2e}')
 
# prettify
fig.subplots_adjust(wspace=0.01)
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
ax[0].set_ylabel('log(power)')
ax[0].set_xlabel(r'$\omega$ (mHz)')
ax[1].set_xlabel(r'$\omega$ (mHz)')
ax[0].set_title(r'Raw Data')
ax[1].set_title(r'Windowed Data')
    
### plot the spectra for one kx = ky = k = 0

# find the indices at this value 
index = np.where(np.abs(fx) == 0)[0]


# loop through omega 
power_kn_windowless = [] 
power_kn_windowed = [] 
for w in range(len(power_windowless)): 
    
    # get the average power over the annulus we are looking at 
    power_windowless_k_annulus = np.log(np.average(power_windowless[w, index, index]))
    
    # do the same thing for the windowed power 
    power_windowed_k_annulus = np.log(np.average(power_windowed[w, index, index])) 

    # take an average of this and store in an array 
    power_kn_windowless.append(power_windowless_k_annulus)
    power_kn_windowed.append(power_windowed_k_annulus)

# plot the windowed and unwindowed data 
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax[0].plot(f_t, power_kn_windowless)
ax[1].plot(f_t, power_kn_windowed)

# prettify 
fig.subplots_adjust(wspace=0.01)
ax[0].set_ylabel('log(power)')
ax[0].set_xlabel(r'$\omega$ (mHz)')
ax[1].set_xlabel(r'$\omega$ (mHz)')
ax[0].set_title(r'Raw Data at $k_x = k_y = k = 0$ 1/m')
ax[1].set_title(r'Windowed Data at $k_x = k_y = k = 0 1/m$')