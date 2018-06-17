# hod

This repository contains code for applying the HOD method described in Smith et al. 2017 (https://arxiv.org/abs/1701.06581)
to populate a halo lightcone with galaxies.

For testing purposes, a small section of the MXXL halo lightcone is provided in input/halo_catalogue_small.hdf5. This halo
catalogue covers an area of 25 sq deg (0<ra<5 and 0<dec<5), to z=1. The full sky halo catalogue to z=2.2 is available at

http://icc.dur.ac.uk/data/

https://tao.asvo.org.au/tao/

## make_catalogue.py

Contains the main program. Running this will read in the small input halo catalogue, populate it with galaxies using the
HOD prescription of Smith+17, assigning each galaxy an r-band magnitude, and g-r colour. The output galaxy 
catalogue will be stored in the file output/cat.hdf5

### Input file properties

Properties stored in input/halo_cataloge_small.hdf5 are described below. Not all of these properties are used in the code.
Properties that are needed are indicated in bold, while properties that are not used are in italics.

- *M200c*: M200crit, the halo mass in spheres with average density 200 times the critical density, in units 1e10 Msun/h

- **M200m**: M200mean, the halo mass in spheres with average density 200 times the mean density, in units 1e10 Msun/h

- **dec**: Declination, in degrees

- **ra**: Right ascension, in degrees

- **rvmax**: Radius at which Vmax occurs, in (comoving) Mpc/h

- *vdisp*: Velocity dispersion, in (proper) km/s

- **vmax**: Vmax, the maximum circular velocity, in (proper) km/s

- **z_cos**: Cosmological redshift, which ignores the peculiar velocity

- **z_obs**: Observed redshift, which takes into account the peculiar velocity

### Output file properties
 
The default properties stored in the output files. The exact properties to store can be specified (see galaxy_catalogue.py)

- 

## parameters.py

## halo_catalogue.py

## galaxy_catalogue.py

## hod.py

## luminosity_function.py

## colour.py

## k_correction.py

##

The class HOD_BGS in hod.py contains methods which return the evolving HODs used to create the MXXL mock
catalogue, and also methods for randomly sampling magnitudes from these HODs. Note that the very first time
an instance of this class is created, large lookup tables of central and satellite galaxy magnitudes are 
created and saved to file, which takes some time. The class HOD_BGS will read from these files once they
are created, which is much faster.
