# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:14:28 2021

@author: zofkm7as
"""

import pandas as pd
import numpy as np
from scipy import interpolate, ndimage
from math import factorial
import matplotlib.pyplot as plt

bend_threshold = 0.0021

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

def extract_bends(x, smooth_y, particle):
	# Find extrema
	ex = (np.diff(np.sign(np.diff(smooth_y))).nonzero()[0] + 1)
	if len(ex)>=2 and ex[0]==0:
		ex = ex[1:]
	bend_times = x[ex]
	bend_magnitude = smooth_y[ex]

	# Sort for extrema satisfying criteria
	idx = np.ones(len(bend_times))
	index = 1
	prev_index = 0
	while index<len(bend_magnitude):
		dist = abs(bend_magnitude[index]-bend_magnitude[prev_index])
		print('particle: {0} - {1}'.format(particle,dist))
		if dist<bend_threshold:
			idx[index] = 0
			if index<len(bend_magnitude)-1:
				idx[index+1] = 0
			index += 2 # look for next maximum/minimum (not just extrema)
		else:
			prev_index = index
			index += 1
	bend_times = bend_times[idx==1]
	return bend_times

df = pd.read_csv('C:/Users/zofkm7as/Downloads/track.csv')

df[['x', 'y', 'particle']].plot(kind='scatter', x='x', y='y', 
                                label='particle')

ax = df.plot(x='x',y='y',kind='scatter',figsize=(8,8))
df[['x','y','particle']].apply(lambda x: ax.text(*x),axis=1)

for particle in list(df['particle'].unique()):
    		# Smooth bend signal
	x = np.arange(df.loc[df['particle'] == particle, 'frame'].min(),
               df.loc[df['particle'] == particle, 'frame'].max())
	f = interpolate.interp1d(df.loc[df['particle'] == particle, 
                                  'frame'].values, 
                                 df.loc[df['particle'] == particle, 
                                           'eccentricity'].values)
	y = f(x)
	smooth_y = savitzky_golay(y, 7, 2)
        
    # Bends
	bend_times = extract_bends(x, smooth_y, particle)
	#print(bend_times)


