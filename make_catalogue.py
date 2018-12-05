#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from halo_catalogue import MXXLCatalogue
from galaxy_catalogue import BGSGalaxyCatalogue
from hod import HOD_BGS
from colour import Colour  
from k_correction import GAMA_KCorrection
import parameters as par


def main():

    import warnings
    warnings.filterwarnings("ignore")

    # create halo catalogue
    halo_cat = MXXLCatalogue()

    # empty galaxy catalogue
    gal_cat  = BGSGalaxyCatalogue(halo_cat)

    # use hods to populate galaxy catalogue
    hod = HOD_BGS()
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    gal_cat.position_galaxies()

    # add g-r colours
    col = Colour()
    gal_cat.add_colours(col)

    # use colour-dependent k-correction to get apparent magnitude
    kcorr = GAMA_KCorrection()
    gal_cat.add_apparent_magnitude(kcorr)

    # cut to galaxies brighter than apparent magnitude threshold
    gal_cat.cut(gal_cat.get("app_mag") <= par.mag_faint)

    # save catalogue to file
    gal_cat.save_to_file(par.output_dir+"cat.hdf5", format="hdf5",
                         halo_properties=["mass",])
    

if __name__ == "__main__":
    main()
