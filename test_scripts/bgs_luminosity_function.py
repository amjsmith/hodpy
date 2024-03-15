import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from hodpy import lookup

# Use LuminosityFunctionTabulated for the BGS tabulated LF
from hodpy.luminosity_function import LuminosityFunctionTabulated

# The BGS LF measurements are E-corrected
# To return the E-corrected LF, set evolution parameters to zero
P = 0
Q = 0

# Luminosity function measured in the North, using Vmax method
# Galaxies in range 0.01 < z < 0.5, with E-correction 0.67*(z-0.1)
lf_file_N = lookup.path+'/bgs/bgs_N_cumulative_lf.dat'
lf_N = LuminosityFunctionTabulated(lf_file_N, P, Q)

# Luminosity function measured in the South, using Vmax method
# Galaxies in range 0.01 < z < 0.5, with E-correction 0.67*(z-0.1)
lf_file_S = lookup.path+'/bgs/bgs_S_cumulative_lf.dat'
lf_S = LuminosityFunctionTabulated(lf_file_S, P, Q)

# Number densities measured from volume-limited samples used in HOD
# fitting. LF is constructed from interpolating between samples.
lf_file = lookup.path+'/bgs/bgs_samples_cumulative_lf.dat'
lf = LuminosityFunctionTabulated(lf_file, P, Q)


# make plot of cumulative LF
mags = np.arange(-25,-12,0.01)
z = 0.2
plt.plot(mags, lf_N.Phi_cumulative(mags, z), label='BGS North')
plt.plot(mags, lf_S.Phi_cumulative(mags, z), label='BGS South')
plt.plot(mags, lf.Phi_cumulative(mags, z), label='Vol lim samples')

plt.legend(loc='upper left').draw_frame(False)

plt.yscale('log')
plt.ylabel(r'$\phi(<M_r) \ / \ \mathrm{Mpc}^{-3}h^3$')
plt.xlabel(r'$M_r$')
plt.show()


# make plot of differential LF
mags = np.arange(-25,-12,0.0001)
z = 0.2
plt.plot(mags, lf_N.Phi(mags, z), label='BGS North')
plt.plot(mags, lf_S.Phi(mags, z), label='BGS South')
plt.plot(mags, lf.Phi(mags, z), label='Vol lim samples')

print(lf_S.Phi(mags, z))

plt.legend(loc='upper left').draw_frame(False)

plt.yscale('log')
plt.ylabel(r'$\phi(M_r) \ / \ \mathrm{Mpc}^{-3}h^3\mathrm{mag}^{-1}$')
plt.xlabel(r'$M_r$')
plt.show()
