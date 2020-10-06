#! /usr/bin/env python
from __future__ import print_function
import numpy as np

from hodpy.halo_catalogue import MXXLSnapshot
from hodpy.galaxy_catalogue_snapshot import BGSGalaxyCatalogueSnapshot
from hodpy.mass_function import MassFunctionMXXL
from hodpy.hod_bgs import HOD_BGS
from hodpy.colour import Colour
from hodpy import lookup


def main(input_file, output_file, snapshot, mag_faint):

    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    halo_cat = MXXLSnapshot(input_file, snapshot)

    # empty galaxy catalogue
    gal_cat  = BGSGalaxyCatalogueSnapshot(halo_cat)

    # use hods to populate galaxy catalogue
    hod = HOD_BGS()
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    gal_cat.position_galaxies()

    # add g-r colours
    col = Colour()
    gal_cat.add_colours(col)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
if __name__ == "__main__":
    
    input_file = "input/snapshot_58_small.hdf5"
    output_file = "output/galaxy_catalogue_snapshot.hdf5"
    snapshot = 58
    mag_faint = -18.0 # faintest absolute magnitude
    
    main(input_file, output_file, snapshot, mag_faint)
