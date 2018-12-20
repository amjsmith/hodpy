#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from halo_catalogue import MXXLSnapshot
from galaxy_catalogue_snapshot import BGSGalaxyCatalogueSnapshot
from hod_bgs import HOD_BGS
from colour import Colour  
from k_correction import GAMA_KCorrection
import parameters as par


def main():

    import warnings
    warnings.filterwarnings("ignore")

    # create halo catalogue
    halo_cat = MXXLSnapshot()

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
    gal_cat.cut(gal_cat.get("abs_mag") <= -20.0)
    
    # save catalogue to file
    gal_cat.save_to_file(par.output_dir+"cat_snapshot.hdf5", format="hdf5",
                         halo_properties=["mass",])
    

if __name__ == "__main__":
    main()
