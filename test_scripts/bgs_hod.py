import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy import lookup

hod = HOD_BGS(cosmology=None, abs_mag_faint=-18,
              hod_param_file=lookup.abacus_hod_parameters, 
              central_lookup_file='test_cen.npy',
              satellite_lookup_file='test_sat.npy',
              replace_central_lookup=True, replace_satellite_lookup=True)



log_mass = np.arange(11,15,0.01)
mass = 10**log_mass

mags = np.arange(-22,-17.9,0.5)
for i in range(len(mags)):
    magnitude = np.ones(len(log_mass)) * mags[i]

    Ncen = hod.number_centrals_mean(log_mass, magnitude)
    plt.plot(mass, Ncen, c="C%i"%i, ls="--")

    Nsat = hod.number_satellites_mean(log_mass, magnitude)
    plt.plot(mass, Nsat, c="C%i"%i, ls=":")

    Ngal = hod.number_galaxies_mean(log_mass, magnitude)
    plt.plot(mass, Ngal, c="C%i"%i)

plt.xscale('log')
plt.yscale('log')

plt.ylim(1e-3,1e3)
plt.show()
