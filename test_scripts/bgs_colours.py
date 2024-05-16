import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from hodpy.colour import ColourDESI

print("Make plot showing the red and blue sequence, and fraction of blue galaxies, at z=0.1")

# colour distributions in the South
col = ColourDESI(photsys='S')

z = 0.1 # redshift to make plot

magnitude = np.arange(-24,-10,0.01)
redshift = np.ones(len(magnitude))*z

red_mean = col.red_mean(magnitude, redshift)
red_rms = col.red_rms(magnitude, redshift)
plt.plot(magnitude, red_mean, c="C3", label="Red mean")
plt.fill_between(magnitude, red_mean-red_rms, red_mean+red_rms,
                 facecolor="C3", alpha=0.3)

blue_mean = col.blue_mean(magnitude, redshift)
blue_rms = col.blue_rms(magnitude, redshift)
plt.plot(magnitude, blue_mean, c="C0", label="Blue mean")
plt.fill_between(magnitude, blue_mean-blue_rms, blue_mean+blue_rms,
                 facecolor="C0", alpha=0.3)

fraction_blue = col.fraction_blue(magnitude, redshift)
plt.plot(magnitude, fraction_blue, c="C0", ls="--", label="Blue fraction")

plt.title('z = %.2f'%z)
plt.legend(loc='upper right').draw_frame(False)
plt.ylim(0,1.2)
plt.xlabel('Mr')
plt.show()


##################################################
print("Make plot showing the fraction of central galaxies at z=0.1")

# To plot the fraction of central galaxies, need to initialize the ColourDESI class
# with the HOD to use
from hodpy.hod_bgs_abacus import HOD_BGS
hod = HOD_BGS(cosmo=0, photsys='S', redshift_evolution=True)
col = ColourDESI(photsys='S', hod=hod)

z = 0.1 # redshift to make plot
magnitude = np.arange(-24,-10,0.01)
redshift = np.ones(len(magnitude))*z

fraction_central = col.fraction_central(magnitude, redshift)
plt.plot(magnitude, fraction_central, c="k", ls="--", label="Central fraction")

plt.title('z = %.2f'%z)
plt.legend(loc='upper right').draw_frame(False)
plt.ylim(0,1.2)
plt.xlabel('Mr')
plt.show()




##################################################
print("Make plot showing the double-Gaussian colour distribution z=0.2")
from scipy.stats import norm

# To plot the fraction of central galaxies, need to initialize the ColourDESI class
# with the HOD to use
col_S = ColourDESI(photsys='S')
col_N = ColourDESI(photsys='N')

z = 0.2 # redshift to make plot
M = -21 # magnitude to make plot

plt.title(r'$z=%.1f, M_r=%.1f$'%(z, M))

magnitude = np.ones(1)*M
redshift = np.ones(1)*z
colour = np.arange(0,1.3,0.01)

# plot in the South
mu_blue = col_S.blue_mean(magnitude, redshift)
sig_blue = col_S.blue_rms(magnitude, redshift)
mu_red = col_S.red_mean(magnitude, redshift)
sig_red = col_S.red_rms(magnitude, redshift)

blue_sequence_south = norm(loc=mu_blue, scale=sig_blue)
red_sequence_south = norm(loc=mu_red, scale=sig_red)

frac_blue_south = col_S.fraction_blue(magnitude, redshift)

plt.plot(colour, blue_sequence_south.pdf(colour)*frac_blue_south, c='C0', ls=":", label='South')
plt.plot(colour, red_sequence_south.pdf(colour)*(1-frac_blue_south), c='C3', ls=":")


# plot in the North
mu_blue = col_N.blue_mean(magnitude, redshift)
sig_blue = col_N.blue_rms(magnitude, redshift)
mu_red = col_N.red_mean(magnitude, redshift)
sig_red = col_N.red_rms(magnitude, redshift)

blue_sequence_north = norm(loc=mu_blue, scale=sig_blue)
red_sequence_north = norm(loc=mu_red, scale=sig_red)

frac_blue_north = col_N.fraction_blue(magnitude, redshift)

plt.plot(colour, blue_sequence_north.pdf(colour)*frac_blue_north, c='C0', ls="--", label='North')
plt.plot(colour, red_sequence_north.pdf(colour)*(1-frac_blue_north), c='C3', ls="--")

# weighted combination
area_south = 5193.6896 
area_north = 2279.0288 
area_fraction = area_south / (area_north+area_south)
plt.plot(colour, blue_sequence_south.pdf(colour)*frac_blue_south*area_fraction + blue_sequence_north.pdf(colour)*frac_blue_north*(1-area_fraction), label='Weighted combination', c='k', ls="-")
plt.plot(colour, red_sequence_south.pdf(colour)*(1-frac_blue_south)*area_fraction + red_sequence_north.pdf(colour)*(1-frac_blue_north)*(1-area_fraction), c='k', ls="-")

plt.xlabel(r'$^{0.1}(g-r)$')
plt.ylabel(r'$N_\mathrm{gal}$')
plt.legend(loc='upper left').draw_frame(False)
plt.show()
