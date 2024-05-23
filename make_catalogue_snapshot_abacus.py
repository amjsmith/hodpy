#! /usr/bin/env python
from __future__ import print_function
import numpy as np

from hodpy.halo_catalogue import AbacusSnapshot
from hodpy.galaxy_catalogue_snapshot import GalaxyCatalogueSnapshotAbacus
from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy.colour import ColourDESI
from hodpy import lookup


def main(input_file, output_file, snapshot, mag_faint    cosmo, snapshot_redshift=0.2, mag_faint=-18):

    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    halo_cat = AbacusSnapshot(file_name, cosmo=cosmo, snapshot_redshift=snapshot_redshift)

    # empty galaxy catalogue
    gal_cat  = GalaxyCatalogueSnapshot(halo_cat, cosmo)

    # use hods to populate galaxy catalogue
    hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='absolute', mag_faint=mag_faint, redshift_evolution=False)
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    gal_cat.position_galaxies()

    # add g-r colours
    col = ColourDESI(photsys=photsys)
    gal_cat.add_colours(col)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])
    
    
    
if __name__ == "__main__":

    
    
    main(input_file, output_file, snapshot, mag_faint)
