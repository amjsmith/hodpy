import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from hodpy.mass_function import MassFunctionAbacus

# plot fit to AbacusSummit halo mass function at a few different redshifts
mf = MassFunctionAbacus(cosmo=0)

log_mass = np.arange(10,16,0.01) #halo mass bins

zs = np.arange(0, 0.51, 0.1) # redshifts to plot
# note that the lowest redshift of AbacusSummit is z=0.1, so the z=0 curve is extrapolated

for i in range(len(zs)):
    z = np.ones(len(log_mass))*zs[i]

    n = mf.number_density(log_mass, z)
    
    plt.plot(10**log_mass, n, label='z = %.1f'%zs[i])
    
plt.legend(loc='upper right').draw_frame(False)
    
plt.xscale('log')
plt.yscale('log')

plt.xlim(1e11,5e15)
plt.ylim(1e-7,1e-1)

plt.xlabel(r'$M \ / \ h^{-1}\mathrm{M}_\odot$')
plt.ylabel(r'$n \ / \ h^{3}\mathrm{Mpc}^{-3}$')

plt.show()
