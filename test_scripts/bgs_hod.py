import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy import lookup

#### Plot original HODs, no redshift evolution

hod = HOD_BGS(cosmo=0, photsys='S', redshift_evolution=False)

log_mass = np.arange(11,15,0.01)
mass = 10**log_mass

mags = np.arange(-22,-17.9,0.5)
for i in range(len(mags)):
    magnitude = np.ones(len(log_mass)) * mags[i]

    Ncen = hod.number_centrals_mean(log_mass, magnitude)
    # plt.plot(mass, Ncen, c="C%i"%i, ls="--")

    Nsat = hod.number_satellites_mean(log_mass, magnitude)
    # plt.plot(mass, Nsat, c="C%i"%i, ls=":")

    Ngal = hod.number_galaxies_mean(log_mass, magnitude)
    plt.plot(mass, Ngal, c="C%i"%i, label=r'$M_r<%.1f$'%mags[i])

plt.title('Original HODs')
plt.legend(loc='upper left').draw_frame(False)

plt.xscale('log')
plt.yscale('log')

plt.ylim(1e-3,1e3)
plt.show()


#### Plot HODs at z=0.5, evolved to match the target luminosity function

hod = HOD_BGS(cosmo=0, photsys='S', redshift_evolution=True)

log_mass = np.arange(11,15,0.01)
mass = 10**log_mass

redshift = np.ones(len(log_mass)) * 0.5

mags = np.arange(-22,-17.9,0.5)
for i in range(len(mags)):
    magnitude = np.ones(len(log_mass)) * mags[i]

    Ncen = hod.number_centrals_mean(log_mass, magnitude, redshift)
    # plt.plot(mass, Ncen, c="C%i"%i, ls="--")

    Nsat = hod.number_satellites_mean(log_mass, magnitude, redshift)
    # plt.plot(mass, Nsat, c="C%i"%i, ls=":")

    Ngal = hod.number_galaxies_mean(log_mass, magnitude, redshift)
    plt.plot(mass, Ngal, c="C%i"%i, label=r'$M_r<%.1f$'%mags[i])

plt.title('HODs evolved to z=0.5')
plt.legend(loc='upper left').draw_frame(False)

plt.xscale('log')
plt.yscale('log')

plt.ylim(1e-3,1e3)
plt.show()
