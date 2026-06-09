#! /usr/bin/env python
import numpy as np
from cut_sky_tools import *
from make_catalogue_snapshot_abacus import main as make_box



if __name__ == '__main__':
    # various parameters to set
    cosmo = 0
    phase = 0
    snapshot_redshift = 0.2
    photsys='S'

    mag_faint_snapshot  = -18 # faintest absolute magitude when populating snapshot
    mag_faint_lightcone = -10 # faintest possible absolute magnitude when populating low-z faint lightcone
    app_mag_faint = 20.2 # apparent magnitude limit of final cut-sky mock

    zmax = 0.6 #0.8      # maximum redshift of lightcone
    zmax_low = 0.15 # maximum redshift of low-z faint lightcone
    mass_cut = 11   # mass cut between unresolved+resolved low z lightcone

    cosmology = CosmologyAbacus(cosmo)
    rmax = cosmology.comoving_distance(zmax)
    rmax_low = cosmology.comoving_distance(zmax_low)

    observer=(-1000,-1000,-1000) #coordinates of observer in the box. This is the corner

    Nfiles = 34
    Lbox = 2000. # box size (Mpc/h)
    SODensity=304.64725494384766

    # location of AbacusSummit simulations
    abacus_path = '/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c%03d_ph%03d/halos/z%.3f/'%(cosmo,phase,snapshot_redshift)

    # path to save outputs
    output_path = '/pscratch/sd/a/amjsmith/AbacusSummit/secondgen_new/z%.3f/AbacusSummit_base_c%03d_ph%03d/new/'%(snapshot_redshift,cosmo,phase)
    
    Nparticle       = lookup.Nparticle_file.format(cosmo,0) #use the same file calculated from ph000
    Nparticle_shell = lookup.Nparticle_shell_file.format(cosmo,0)

    # file names
    halo_snapshot_file   = abacus_path+'halo_info/halo_info_%03d.asdf' 
    galaxy_snapshot_file = output_path+'BGS_box_%s'%(photsys) + '_%03d.fits' 
    halo_lightcone_unres = output_path+'halo_lightcone_unresolved_%03d.hdf5'
    galaxy_lightcone_unres = output_path+'galaxy_lightcone_unresolved_%s'%(photsys) + '_%03d.fits'
    galaxy_lightcone_res   = output_path+'galaxy_lightcone_resolved_%s'%(photsys) + '_%03d.fits'
    galaxy_cutsky_low      = output_path+'galaxy_cut_sky_low_%s'%(photsys) + '_%03d.fits'
    galaxy_cutsky          = output_path+'galaxy_cut_sky_%s'%(photsys) + '_%03d.hdf5'

    galaxy_cutsky_final    = output_path+'galaxy_cut_sky_%s.fits'%(photsys)



    
    print("POPULATING CUBIC BOX")

    for file_number in range(Nfiles):
        input_file = halo_snapshot_file%file_number
        output_file = galaxy_snapshot_file%file_number
        
        # populate the haloes with galaxies
        make_box(input_file, output_file, cosmo=cosmo, photsys=photsys, snapshot_redshift=snapshot_redshift, mag_faint=mag_faint_snapshot)



    print("MAKING LOW REDSHIFT LIGHTCONE OF UNRESOLVED HALOES")

    # this function will loop through the 34 particle files, finding minimum halo mass needed to make mock to faint app mag limit 
    # (taking into account scaling of magnitudes by cosmology)
    # This requires counting the available number of field particles in shells, which is saved to a file.
    # If the files don't exist yet, they will be automatically created (but this is fairly slow)
    # app mag limit is also shifted slightly fainter than is needed, to be safe

    #finding the number of particles in shells is unaffected by the HOD.
    #should be changed for different simulations, but should be fine to use the same file for c000 ph000-024
    output_file = halo_lightcone_unres

    halo_lightcone_unresolved(output_file, path='/global/cfs/cdirs/desi/cosmosim/Abacus/', cosmo=cosmo, ph=phase, photsys=photsys,
                              snapshot_redshift=snapshot_redshift, Nparticle=Nparticle, Nparticle_shell=Nparticle_shell,
                              box_size=Lbox, SODensity=SODensity, observer=observer, 
                              app_mag_faint=app_mag_faint+0.05, Nfiles=Nfiles)


  
    print("MAKING UNRESOLVED LOW Z GALAXY LIGHTCONE")

    # this will populate the unresolved halo lightcone with galaxies

    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
        input_file = halo_lightcone_unres%file_number
        output_file = galaxy_lightcone_unres%file_number

        main_unresolved(input_file, output_file, cosmo=cosmo, photsys=photsys, snapshot_redshift=snapshot_redshift, 
                    mag_faint=mag_faint_lightcone,
                    box_size=Lbox, SODensity=SODensity,
                    zmax=zmax_low+0.01, log_mass_max=mass_cut)


    print("MAKING RESOLVED LOW Z GALAXY LIGHTCONE")

    # this will populate the resolved haloes in the lightcone with faint galaxies,
    # for making the lightcone at low redshifts

    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
    
        input_file = halo_snapshot_file%file_number
        output_file = galaxy_lightcone_res%file_number

        # for a small box, periodic replications would be needed here
        make_box(input_file, output_file, cosmo=cosmo, photsys=photsys, snapshot_redshift=snapshot_redshift, box_size=Lbox,
                 mag_faint=mag_faint_lightcone, rmax=rmax_low+100, log_mass_min=mass_cut, observer=observer, replication=(0,0,0))



    

    print("MAKE CUT-SKY FROM LOW Z LIGHTCONE")

    hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='absolute', mag_faint=-10, redshift_evolution=False,
                  replace_central_lookup=True, replace_satellite_lookup=True)

    # make cut-sky mock, with evolving LF, from faint, low z lightcone
    # this will use the magnitudes rescaled to match target LF exactly
    # note that the low z resolved/unresolved lightcones are already shifted 
    # to have observer at origin

    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)

        unresolved_file = galaxy_lightcone_unres%file_number
        resolved_file = galaxy_lightcone_res%file_number
    
        output_file = galaxy_cutsky_low%file_number

        if os.path.isfile(resolved_file):
            make_lightcone_lowz(resolved_file, unresolved_file, output_file, hod, photsys, app_mag_faint,
                                snapshot_redshift=snapshot_redshift, box_size=Lbox, zmax=zmax_low)


    


    print("MAKE CUT-SKY FROM SNAPSHOT")

    # make cut-sky mock, with evolving LF, from the snapshot files
    # this will use the magnitudes rescaled to match target LF exactly

    for file_number in range(Nfiles):
        print("FILE NUMBER", file_number)
    
        input_file = galaxy_snapshot_file%file_number
        output_file = galaxy_cutsky%file_number

        make_lightcone(input_file, output_file, hod, photsys, mag_faint=app_mag_faint+0.05, 
                       snapshot_redshift=snapshot_redshift, box_size=Lbox, observer=observer, zmax=zmax)
    


    join_files(galaxy_cutsky, galaxy_cutsky_low, output_file=galaxy_cutsky_final, zmax_low=zmax_low, zmax=zmax, app_mag_faint=app_mag_faint)
