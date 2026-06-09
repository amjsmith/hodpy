import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from hodpy.k_correction import DESI_KCorrection, DESI_KCorrection_color 



#Plot the r-band k-corrections as a function of z in the north and south

kcorr_r_N = DESI_KCorrection(band='r', photsys='N') #North
kcorr_r_S = DESI_KCorrection(band='r', photsys='S') #South

z = np.arange(0, 0.6,0.01)
colours = np.arange(0.3,1.1,0.15)

for i in range(len(colours)):
    c = np.ones(len(z)) * colours[i]
    kr_N = kcorr_r_N.k(z, c)
    kr_S = kcorr_r_S.k(z, c)
    
    plt.plot(z, kr_N, c="C%i"%i, label=r'${}^{0.1}(g-r) = %.2f$'%colours[i])
    plt.plot(z, kr_S, c="C%i"%i, ls="--")

plt.plot([],[],c="k",label="North")
plt.plot([],[],c="k",ls="--",label="South")
    
plt.legend(loc='upper left').draw_frame(False)
plt.xlabel(r'$z$')
plt.ylabel(r'$k_r(z)$')
plt.show()




# Converting from observer-frame to rest-frame colours and back again

# Due to using look up tables, we don't get the exact same colour back again

kcorr_col_N = DESI_KCorrection_color(photsys='N')

# use a uniform distribution of colours at z=0.2
obs_col = np.arange(0.4,1.2,0.001)
z = np.ones(len(obs_col)) * 0.2

# convert to rest-frame colours
rest_col = kcorr_col_N.rest_frame_colour(z, obs_col)

# convert rest-frame back to observed colours
obs_col2 = kcorr_col_N.observer_frame_colour(z, rest_col)

# plot the histogram of colours
bin_width = 0.02
bins = np.arange(0,1.5,bin_width)
hist, bins = np.histogram(obs_col, bins=bins)
plt.plot(bins[:-1]+bin_width/2., hist, label=r'observed $g-r$')

hist, bins = np.histogram(rest_col, bins=bins)
plt.plot(bins[:-1]+bin_width/2., hist, label=r'rest-frame ${}^{0.1}(g-r)$')

hist, bins = np.histogram(obs_col2, bins=bins)
plt.plot(bins[:-1]+bin_width/2., hist, label=r'new $g-r$')

plt.legend(loc='upper left').draw_frame(False)
plt.xlabel("g-r")
plt.ylabel("N")
plt.show()
