import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from hodpy.colour import ColourDESI


# colour distributions in the South
col = ColourDESI(photsys='S')

z = 0.05 # redshift to make plot


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


plt.legend(loc='upper right').draw_frame(False)

plt.ylim(0,1.2)

plt.xlabel('Mr')

plt.show()
