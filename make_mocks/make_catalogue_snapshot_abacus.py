#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from astropy.table import Table, vstack

from hodpy.halo_catalogue import AbacusSnapshot
from hodpy.galaxy_catalogue_snapshot import BGSGalaxyCatalogueSnapshotAbacus
from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy.colour import ColourDESI
from hodpy.k_correction import DESI_KCorrection
from hodpy import lookup



def main(input_file, output_file, cosmo, photsys, snapshot_redshift=0.2, mag_faint=-18,
         rmax=None, log_mass_min=None, log_mass_max=None, observer=None, replication=(0,0,0),
         box_size=2000.):
    '''
    Create a cubic box BGS mock
    '''
    import warnings
    warnings.filterwarnings("ignore")
    
    # to be safe, re-make the magnitude lookup tables on first loop iteration
    file_number = int(input_file[-8:-5])
    replace_lookup = file_number==0
    
    # create halo catalogue
    halo_cat = AbacusSnapshot(input_file, cosmo=cosmo, snapshot_redshift=snapshot_redshift)

    # apply halo mass cuts, if needed
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        keep = np.ones(len(log_mass), dtype="bool")
        if not log_mass_min is None:
            keep = np.logical_and(keep, log_mass >= log_mass_min)
        if not log_mass_max is None:
            keep = np.logical_and(keep, log_mass <= log_mass_max)
        halo_cat.cut(keep)
       
    # cut to haloes that are within a comoving distance corresponding to the redshift zmax
    # this is done is we are making a lightcone, to remove any high redshift haloes we don't need
    # Note that if we are making a lightcone, the output file of this function will still be
    # in Cartesian coordinates, and will need to be converted to cut-sky
    if not rmax is None:
        print("cutting to rmax")
        pos = halo_cat.get("pos")
        
        # make sure observer is at origin
        for i in range(3):
            pos[:,i] -= observer[i]
            
        #apply periodic boundary conditions, so -Lbox/2 < pos < Lbox/2
        #if apply_periodic==True:
        pos[pos>box_size/2.]-=box_size
        pos[pos<-box_size/2.]+=box_size
        
        # replicate the box
        pos[:,0] += replication[0]*box_size
        pos[:,1] += replication[1]*box_size
        pos[:,2] += replication[2]*box_size
        
        halo_cat.add("pos", pos)
        
        dist = np.sum(pos**2, axis=1)**0.5
        #dist_max = cosmology.comoving_distance(np.array([zmax,]))[0]
        halo_cat.cut(dist<rmax)
        
        if len(halo_cat.get("zcos")) == 0:
            print("No haloes in lightcone, skipping file")
            return
    
    # empty galaxy catalogue
    gal_cat  = BGSGalaxyCatalogueSnapshotAbacus(halo_cat, cosmo)

    # use hods to populate galaxy catalogue
    hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='absolute', mag_faint=mag_faint, redshift_evolution=False,
                  z0=snapshot_redshift, replace_central_lookup=replace_lookup, replace_satellite_lookup=replace_lookup)
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    gal_cat.position_galaxies()
    
    # add g-r colours
    col = ColourDESI(photsys=photsys, hod=hod)
    gal_cat.add_colours(col)

    # cut to galaxies brighter than absolute magnitude threshold
    gal_cat.cut(gal_cat.get("abs_mag") <= mag_faint)
    
    # save catalogue to file
    gal_cat.save_to_file(output_file, format="fits_BGS")
    
    
def join_files(path, photsys):
    '''
    Combine the cubic box outputs into a single file
    '''
    for file_number in range(34):
        print(file_number)

        table_i = Table.read(path+'BGS_box_%s_%03d.fits'%(photsys,file_number))
        table_i['FILE_NUM'][:] = file_number

        #table_i = table_i['R_MAG_ABS', 'G_R_REST', 'HALO_MASS', 'cen', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'FILE_NUM', 'HALO_ID']

        if file_number==0:
            table = table_i
        else:
            table = vstack([table, table_i])

        del table_i
    
    # write the new table
    table.write(output_path+'BGS_box_%s.fits'%(photsys), format="fits")
    
        
    
    
if __name__ == "__main__":

    cosmo = 0
    phase = 0
    snapshot_redshift = 0.2
    photsys='S'
    mag_faint = -18

    # location of the snapshots on NERSC
    abacus_path = '/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c%03d_ph%03d/halos/z%.3f/'%(cosmo,phase,snapshot_redshift)
    
    output_path = '/pscratch/sd/a/amjsmith/AbacusSummit/secondgen_new/z%.3f/AbacusSummit_base_c%03d_ph%03d/'%(snapshot_redshift,cosmo,phase)
    
    # each snapshot is split into 34 files
    for file_number in range(34):
        input_file = abacus_path+'halo_info/halo_info_%03d.asdf'%file_number
        output_file = output_path+'BGS_box_%s_%03d.fits'%(photsys, file_number)
        
        # populate the haloes with galaxies
        main(input_file, output_file, cosmo=cosmo, photsys=photsys, snapshot_redshift=snapshot_redshift, mag_faint=mag_faint)

    # join the 34 outputs into a single file
    join_files(output_path, photsys)
