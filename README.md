# hod

This repository contains code for applying the HOD method described in Smith et al. 2017 to populate a 
halo lightcone with galaxies

The class HOD_BGS in hod.py contains methods which return the evolving HODs used to create the MXXL mock
catalogue, and also methods for randomly sampling magnitudes from these HODs. Note that the very first time
an instance of this class is created, large lookup tables of central and satellite galaxy magnitudes are 
created and saved to file, which takes some time. The class HOD_BGS will read from these files once they
are created, which is much faster.
