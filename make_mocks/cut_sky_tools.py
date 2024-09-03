import numpy as np
from os.path import exists
from abacusnbody.data.read_abacus import read_asdf
import gc
import os
import h5py
from astropy.table import Table, vstack

import sys
sys.path.append('..')
from hodpy import lookup
from hodpy.cosmology import CosmologyAbacus
from hodpy.hod_bgs_abacus import HOD_BGS
from hodpy.mass_function import MassFunctionAbacus
from hodpy.k_correction import DESI_KCorrection, DESI_KCorrection_color
from hodpy.luminosity_function import LuminosityFunctionTabulated
from hodpy.catalogue import Catalogue
from hodpy.halo_catalogue import AbacusSnapshotUnresolved
from hodpy.galaxy_catalogue_snapshot import BGSGalaxyCatalogueSnapshotAbacus
from hodpy.colour import ColourDESI




### For converting cartesian coordinates to a cut-sky mock

def cut_sky(gal_cat, photsys, hod, Lbox=2000., replication=(0,0,0), zcut=None, mag_cut=None, z0=0.2):
    """
    Creates a cut sky mock by converting the cartesian coordiantes of a cubic box mock to ra, dec, z.
    Magnitudes and colours are evolved with redshift
    Args:
        position:  array of comoving position vectors (Mpc/h), in the range -Lbox/2 < pos < Lbox/2
        velocity:  array of proper velocity vectors (km/s)
        magnitude: array of absolute magnitude
        is_cen:    boolean array indicating if galaxy is central (True) or satellite (False)
        cosmology: instance of astropy.cosmology class
        Lbox:      comoving box length of simulation (Mpc/h)
        kcorr_r:   GAMA_KCorrection object with r-band k-correction
        kcorr_g:   GAMA_KCorrection object with g-band k-correction
        replication: tuple indicating which periodic replication to use. Default value is (0,0,0) 
                         (ie no replications).
        zcut:    If provided, will only return galaxies with z<=zcut. By default will return
                         all galaxies.
        mag_cut: If provided, will only return galaxies with apparent magnitude < mag_cut. 
                         By default will return all galaxies.
        cosmology_orig: instance of astropy.cosmology class. The original simulation cosmology.
                         If provided, magnitudes will be scaled by cosmology
    Returns:
        ra:   array of ra (deg)
        dec:  array of dec (deg)
        zcos: array of cosmological redshift, which does not include the effect of peculiar velocities
        zobs: array of observed redshift, which includes peculiar velocities.
        magnitude_new: array of new absolute magnitude, rescaled to match target luminosity 
                          function at each redshift
        app_mag: array of apparent magnitudes (calculated from rescaled magnitudes and colours)
        colour_new: array of g-r colours, which are re-assigned to add evolution
        colour_obs: array of observer-frame g-r colours
        index: array of indices. Used to match galaxies between the input and output arrays of 
                  this function
    """
    
    gal_cat['index'] = np.arange(len(gal_cat['x']), dtype=np.int64)
    
    cosmology = hod.cosmo
    cat = Catalogue(cosmology) # use this for converting coordinates to ra, dec, z
    
    position_rep = np.array([gal_cat['x'], gal_cat['y'], gal_cat['z']]).transpose()
    if replication==(0,0,0):
        print("No periodic replications")
    else:
        print("Applying periodic replications")
        for i in range(3):
            print("%.1f < %s < %.1f"%((-1+2*replication[i])*Lbox/2., chr(120+i), (1+2*replication[i])*Lbox/2.))
            position_rep[:,i] += Lbox*replication[i]
    
    gal_cat['x_rep'] = position_rep[:,0]
    gal_cat['y_rep'] = position_rep[:,1]
    gal_cat['z_rep'] = position_rep[:,2]
    
    #print(np.min(position_rep[:,1]), np.max(position_rep[:,1]))
    
    gal_cat['RA'], gal_cat['DEC'], gal_cat['Z_COSMO'] = cat.pos3d_to_equatorial(position_rep)
    velocity = np.array([gal_cat['vx'], gal_cat['vy'], gal_cat['vz']]).transpose()
    vlos = cat.vel_to_vlos(position_rep, velocity)
    gal_cat['Z'] = cat.vel_to_zobs(gal_cat['Z_COSMO'], vlos)
    
    #print(gal_cat['Z'])
    print(np.min(gal_cat['Z']), np.max(gal_cat['Z']))
    
    if not zcut is None:
        print("Applying redshift cut z < %.2f"%zcut)
        keep = gal_cat['Z'] <= zcut
        gal_cat = gal_cat[keep]
                  
    
    print("Assigning colours")
    is_cen = gal_cat['cen']==1
    is_sat = gal_cat['cen']==0
    colour_new = np.zeros(len(is_cen))
    
    col = ColourDESI(photsys=photsys, hod=hod, cutsky=True, cutsky_z0=z0)
    
    # randomly assign colours to centrals and satellites
    colour_new[is_cen] = col.get_central_colour(gal_cat['R_MAG_ABS'][is_cen], gal_cat['Z'][is_cen])
    colour_new[is_sat] = col.get_satellite_colour(gal_cat['R_MAG_ABS'][is_sat], gal_cat['Z'][is_sat])
    gal_cat['G_R_REST'] = colour_new
    
    # get apparent magnitude
    kcorr_r = DESI_KCorrection(band='r', photsys=photsys, cosmology=cosmology) 
    gal_cat['R_MAG_APP'] = kcorr_r.apparent_magnitude(gal_cat['R_MAG_ABS'], gal_cat['Z'], gal_cat['G_R_REST'])
    
    if len(gal_cat['Z']) > 0:
        print(np.min(gal_cat['Z']), np.max(gal_cat['Z']))
        print(np.min(gal_cat['G_R_REST']), np.max(gal_cat['G_R_REST']))
    
    # observer frame colours
    kcorr_col = DESI_KCorrection_color(photsys=photsys)
    gal_cat['G_R_OBS'] = kcorr_col.observer_frame_colour(gal_cat['Z'], np.clip(gal_cat['G_R_REST'],-3.9,3.9))
    
    if not mag_cut is None:
        print("Applying magnitude cut r < %.2f"%mag_cut)
        keep = gal_cat['R_MAG_APP'] <= mag_cut
        gal_cat = gal_cat[keep]
        
    
    return gal_cat


### for getting the number of field particles to use as tracers of unresolved haloes

def num_in_rmax(p, rmax, box_size):
  
    
    # num of times each particle exists in cube of side length 2*rmax, with observer in centre, after applying replications

    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = p.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                keep = np.logical_and.reduce([p_i[:,0] <= rmax, p_i[:,0] >= -rmax,
                                              p_i[:,1] <= rmax, p_i[:,1] >= -rmax,
                                              p_i[:,2] <= rmax, p_i[:,2] >= -rmax])
                
                number += np.count_nonzero(keep)
                
    return number


def num_in_shell(p, rmin, rmax, box_size=2000):
  
    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    rmin2 = rmin**2
    rmax2 = rmax**2
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = p.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                dist2 = np.sum(p_i**2, axis=1)
                
                keep = np.logical_and(dist2>=rmin2, dist2<rmax2)
                
                number += np.count_nonzero(keep)
                
    return number


def particles_in_shell(pos, vel, box_size, rmin, rmax):
    
    nrep = 0
    for i in range(100):
        if rmax >= (box_size*(2*i+1))/2.:
            nrep = i+1
        else: 
            break
            
    number=0
    
    rmin2 = rmin**2
    rmax2 = rmax**2
    
    pos_shell = [None]*(nrep*2+1)**3
    vel_shell = [None]*(nrep*2+1)**3
    idx=0
    
    for i in range(-nrep, nrep+1):
        for j in range(-nrep, nrep+1):
            for k in range(-nrep, nrep+1):
                p_i = pos.copy()
                p_i[:,0] += box_size*i
                p_i[:,1] += box_size*j
                p_i[:,2] += box_size*k
                
                dist2 = np.sum(p_i**2, axis=1)
                
                keep = np.logical_and(dist2>=rmin2, dist2<rmax2)
                
                if np.count_nonzero(keep) > 0:
                    pos_shell[idx] = p_i[keep]
                    vel_shell[idx] = vel[keep]
                else:
                    pos_shell[idx] = np.zeros((0,3))
                    vel_shell[idx] = np.zeros((0,3))
                idx += 1

    pos_shell = np.concatenate(pos_shell)
    vel_shell = np.concatenate(vel_shell)
    
    return pos_shell, vel_shell
            




def halo_lightcone_unresolved(output_file, path, cosmo, ph, photsys, snapshot_redshift,
                              Nparticle, Nparticle_shell, box_size=2000., SODensity=200,
                              simulation="base", observer=(0,0,0), app_mag_faint=20.25, Nfiles=34):
    """
    Create a lightcone of unresolved AbacusSummit haloes, using the field particles 
    (not in haloes) as tracers. The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        mf_fit_file:       path of file of mass function. Will be created if it doesn't exist
        Nparticle:         file containing total number of field particles in each Abacus file.
                                Will be created if it doesn't exist
        Nparticle_shell:   file containing total number of field particles in shells of comoving
                                distance, ineach Abacus file. Will be created if it doesn't exist
        box_size:          float, simulation box size (Mpc/h)
        SODensity:         float, spherical overdensity of L1 haloes
        simulation:        string, the AbacusSummit simulation, default is "base"
        cosmo:             integer, the AbacusSummit cosmology number, default is 0 (Planck LCDM)
        ph:                integer, the AbacusSummit simulation phase, default is 0
        observer:          3D position vector of the observer, in units Mpc/h. By default 
        app_mag_faint:     float, faint apparent magnitude limit
        Nfiles:      Number of AbacusSummit files for this snapshot. Default is 34
    """
    
    import warnings
    warnings.filterwarnings("ignore")
    
    mock = "AbacusSummit_%s_c%03d_ph%03d"%(simulation, cosmo, ph)
    
    mf = MassFunctionAbacus(cosmo)
    
    # read HOD files
    hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='apparent', mag_faint=app_mag_faint, redshift_evolution=False,
                  replace_central_lookup=True, replace_satellite_lookup=True, z0=snapshot_redshift)
    
    # get min mass
    # rcom bins go up to 1000 Mpc/h
    rcom, logMmin = get_min_mass(app_mag_faint, photsys=photsys, hod=hod, cosmology=hod.cosmo)
    
    # get total number of field particles (using A particles)
    
    if exists(Nparticle) and exists(Nparticle_shell):
        print("Reading total number of field particles")
        N = np.loadtxt(Nparticle)
        Nshells = np.loadtxt(Nparticle_shell)
    
    else:
        print("File doesn't exist yet, finding number of field particles")
        N = np.zeros(Nfiles, dtype="i")
        Nshells = np.zeros((Nfiles,len(rcom)),dtype="i")
        for file_number in range(Nfiles):
            # this loop is slow. Is there a faster way to get total number of field particles in each file?
            file_name = path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
            data = read_asdf(file_name, load_pos=True, load_vel=False)
            p = data["pos"]
            for i in range(3):
                p[:,i] -= observer[i]
            p[p>box_size/2.] -= box_size
            p[p<-box_size/2.] += box_size
            
            del data

            rmax = 1000
            N[file_number] = num_in_rmax(p, rmax, box_size)
    
            for j in range(len(rcom)):
                Nshells[file_number,j] = num_in_shell(p, rmin=rcom[j]-25, rmax=rcom[j], box_size=box_size)
                print(file_number, j, Nshells[file_number,j])
            
            gc.collect() # need to run garbage collection to release memory
        
        # save files
        np.savetxt(Nparticle, N)
        np.savetxt(Nparticle_shell, Nshells)
    
    
    # Now make lightcone of unresolved haloes
    for file_number in range(Nfiles):
        file_name = path+mock+"/halos/z%.3f/field_rv_A/field_rv_A_%03d.asdf"%(snapshot_redshift, file_number)
        print(file_name)

        # read file
        data = read_asdf(file_name, load_pos=True, load_vel=True)
        vel = data["vel"]
        pos = data["pos"]
        for i in range(3):
            pos[:,i] -= observer[i]
        pos[pos>box_size/2.] -= box_size
        pos[pos<-box_size/2.] += box_size
        
        pos_bins = [None]*len(rcom)
        vel_bins = [None]*len(rcom)
        mass_bins = [None]*len(rcom)
        
        for j in range(len(rcom)):
            
            rmin_bin, rmax_bin = rcom[j]-25, rcom[j]
            vol_bin = 4/3.*np.pi*(rmax_bin**3 - rmin_bin**3)
            if j==0:
                logMmin_bin, logMmax_bin = logMmin[j], logMmin[-1]
            else:
                logMmin_bin, logMmax_bin = logMmin[j-1], logMmin[-1]
                
            
            # cut to particles in shell
    
            pos_shell, vel_shell = particles_in_shell(pos, vel, box_size, 
                                                      rmin=rmin_bin, rmax=rmax_bin)
            N_shell = pos_shell.shape[0]
            
            print(file_number, j, N_shell)
            
            if N_shell==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
                
            try:
                Npar =  np.sum(Nshells[:,j]) # total number of field particles in shell
            except:
                Npar =  Nshells[j]
            
            # number of randoms to generate in shell
            Nrand = mf.number_density_in_mass_bin(logMmin_bin, logMmax_bin, snapshot_redshift) * vol_bin
            
            if Nrand==0: 
                pos_bins[j] = np.zeros((0,3))
                vel_bins[j] = np.zeros((0,3))
                mass_bins[j] = np.zeros(0)
                continue
            
            #print(Npar, Nrand, np.count_nonzero(keep_shell))
            
            # probability to keep a particle
            prob = Nrand*1.0 / Npar
            #print(prob)
        
            keep = np.random.rand(N_shell) <= prob
            pos_bins[j] = pos_shell[keep]
            vel_bins[j] = vel_shell[keep]
        
            print(np.count_nonzero(keep), logMmin_bin, logMmax_bin)
        
            # generate random masses if number of particles > 0
            if np.count_nonzero(keep)>0:
                mass_bins[j] = 10**mf.get_random_masses(np.count_nonzero(keep), logMmin_bin, logMmax_bin, snapshot_redshift) / 1e10
            else:
                mass_bins[j] = np.zeros(0)
            
        del data
        gc.collect() # need to run garbage collection to release memory
        
        pos_bins = np.concatenate(pos_bins)
        vel_bins = np.concatenate(vel_bins)
        mass_bins = np.concatenate(mass_bins)
        
        # shift positions back
        # for i in range(3):
        #     pos_bins[:,i] += observer[i]
        
        # save halo lightcone file
        f = h5py.File(output_file%file_number, "a")
        f.create_dataset("mass", data=mass_bins, compression="gzip")
        f.create_dataset("position", data=pos_bins, compression="gzip")
        f.create_dataset("velocity", data=vel_bins, compression="gzip")
        f.close()

        

def get_min_mass(rfaint, photsys, hod, cosmology, zsnap=0.2):
    """
    Get the minimum halo masses, in shells of comoving distance, needed to create a lightcone
    down to a faint apparent magnitude limit. This takes into account the rescaling of
    magnitudes by cosmology. For making Abacus mocks that were fit to MXXL, cosmo_orig
    is the MXXL cosmology, and cosmo_new is the Abacus cosmology.
    Args:
        rfaint:     float, minimum r-band apparent magnitude
        hod:        HOD_BGS object
        cosmo_orig: hodpy.cosmology.Cosmology object, the original cosmology
        cosmo_new:  hodpy.cosmology.Cosmology object, the new cosmology
    """
    
    # bins of comoving distance to find minimum masses in
    rcom = np.arange(25,1001,25)
    z = cosmology.redshift(rcom)
    
    kcorr = DESI_KCorrection(band='r', photsys=photsys, cosmology=cosmology)
    lf = LuminosityFunctionTabulated(lookup.bgs_lf_target.format(hod.c,0), P=0, Q=0)

    # absolute magnitude corresponding to rfaint depends on colour
    # calculate for red and blue galaxies, and choose the faintest mag
    mag_faint1 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*-10)
    mag_faint2 = kcorr.absolute_magnitude(np.ones(len(z))*rfaint, z, np.ones(len(z))*10)
    mag_faint = np.maximum(mag_faint1, mag_faint2)
        
        
    # Now get minimum halo masses needed to add central galaxies brighter than the faint abs mag
    log_mass = np.arange(9,15,0.001)
    mass_limit = np.zeros(len(z))

    for i in range(len(z)):
        N = hod.number_centrals_mean(log_mass,np.ones(len(log_mass))*mag_faint[i],np.ones(len(log_mass))*zsnap)
        idx = np.where(N>0)[0][0]
        mass_limit[i] = log_mass[idx]
        
    for i in range(len(mass_limit)):
        mass_limit[i] = np.min(mass_limit[i:])
        
    return rcom, mass_limit




def main_unresolved(input_file, output_file, cosmo, photsys, snapshot_redshift=0.2, mag_faint=-12, 
                    box_size=2000., SODensity=200,
                    zmax=0.6, log_mass_min=None, log_mass_max=None):
    """
    Create a HOD mock catalogue by populating a hdf5 file of unresolved haloes. 
    The output galaxy catalogue is in Cartesian coordinates
    
    Args:
        input_file:        string, containting the path to the hdf5 file of unresolved haloes
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint absolute magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        hod_param_file:    string, path to file containing HOD hyperparameter fits
        central_lookup_file: lookup file of central magnitudes, will be created if the file
                                doesn't already exist
        satellite_lookup_file: lookup file of satellite magnitudes, will be created if the file
                                doesn't already exist
        box_size:          float, simulation box size (Mpc/h)
        zmax:              float, maximum redshift. If provided, will cut the box to only haloes
                                that are within a comoving distance to the observer that 
                                corresponds to zmax. By default, is None
        observer:          3D position vector of the observer, in units Mpc/h. By default 
                                observer is at the origin (0,0,0) Mpc/h
        log_mass_min:      float, log10 of minimum halo mass cut, in Msun/h
        log_mass_max:      float, log10 of maximum halo mass cut, in Msun/h
    """
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # create halo catalogue
    print("read halo catalogue")
    halo_cat = AbacusSnapshotUnresolved(input_file, cosmo=cosmo, snapshot_redshift=snapshot_redshift,
                                        box_size=box_size, SODensity=SODensity)
    
    # apply cuts to halo mass, if log_mass_min or log_mass_max is provided
    # this cut is applied to make sure there no overlap in masses between the unresolved/resolved
    # halo lightcones
    if not log_mass_min is None or not log_mass_max is None:
        log_mass = halo_cat.get("log_mass")
        
        print(np.min(log_mass), np.max(log_mass))
        
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
    if not zmax is None:
        cosmology = CosmologyAbacus(cosmo)
        pos = halo_cat.get("pos")
        
        dist = np.sum(pos**2, axis=1)**0.5
        dist_max = cosmology.comoving_distance(np.array([zmax,]))[0]
        halo_cat.cut(dist<dist_max)
        
        if len(halo_cat.get("zcos")) == 0:
            print("No haloes in lightcone, skipping file")
            return
    
    # generate a halo id
    # negative id means it's unresolved
    file_number = int(input_file[-8:-5])
    halo_id = np.arange(len(halo_cat.get("zcos")),dtype=np.int64) + (file_number*-1e10)
    halo_cat.add('id', halo_id)
    
    # use hods to populate galaxy catalogue
    print("read HODs")
    replace_lookup = file_number==0
    hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='absolute', mag_faint=mag_faint, redshift_evolution=False,
                 replace_central_lookup=replace_lookup, replace_satellite_lookup=replace_lookup)
    
    # empty galaxy catalogue
    print("create galaxy catalogue")
    gal_cat  = BGSGalaxyCatalogueSnapshotAbacus(halo_cat, cosmo)
    
    print("add galaxies")
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    print("position galaxies")
    gal_cat.position_galaxies(conc="conc")

    # add g-r colours
    print("assigning g-r colours")
    col = ColourDESI(photsys=photsys, hod=hod)
    gal_cat.add_colours(col)

    gal_cat.save_to_file(output_file, format="fits_BGS")



def make_lightcone_lowz(resolved_file, unresolved_file, output_file, 
                        hod, photsys, mag_faint, snapshot_redshift=0.2, box_size=2000., 
                        zmax=0.15):
    """
    Create a cut-sky mock (with ra, dec, z), from the cubic box galaxy mock,
    also rescaling magnitudes by cosmology if the original cosmology is provided
    Args:
        resolved_file:     string, containing the path of the input resolved galaxy mock
        unresolved_file:   string, containing the path of the input unresolved galaxy mock
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint apparent magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        box_size:          float, simulation box size (Mpc/h)
        observer:          3D position vector of the observer, in units Mpc/h. By default
        zmax:              float, maximum redshift cut. 
        cosmology_orig:    object of class hodpy.cosmology.Cosmology, if provided, this is the
                                original cosmology when doing cosmology rescaling.
        mag_dataset:       string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    
    # read resolved and unresolved galaxy catalogues
    cat_resolved = Table.read(resolved_file)
    cat_resolved['res'] = np.ones(len(cat_resolved['x']), dtype=np.int32)
    
    try:
        cat_unresolved = Table.read(unresolved_file)
        cat_unresolved['res'] = np.zeros(len(cat_unresolved['x']), dtype=np.int32)
    except:
        cat_unresolved = Table()
        
   
    # combine into single array
    cat = vstack([cat_resolved, cat_unresolved])
    
    cat['box_ind'] = np.ones(len(cat['x']), dtype="i")*-1
    
    cat = cut_sky(cat, photsys, hod, Lbox=box_size, replication=(0,0,0), zcut=zmax, mag_cut=mag_faint)
    
    cat.write(output_file, format="fits")




def make_lightcone(input_file, output_file, hod, photsys, mag_faint, snapshot_redshift=0.2, box_size=2000., observer=(0,0,0), zmax=0.6):
    """
    Create a cut-sky mock (with ra, dec, z), from the cubic box galaxy mock,
    also rescaling magnitudes by cosmology if the original cosmology is provided
    Args:
        input_file:        string, containing the path of the input galaxy mock
        output_file:       string, containing the path of hdf5 file to save outputs
        snapshot_redshift: integer, the redshift of the snapshot
        mag_faint:         float, faint apparent magnitude limit
        cosmology:         object of class hodpy.cosmology.Cosmology, the simulation cosmology
        box_size:          float, simulation box size (Mpc/h)
        observer:          3D position vector of the observer, in units Mpc/h. By default
        zmax:              float, maximum redshift cut. 
        cosmology_orig:    object of class hodpy.cosmology.Cosmology, if provided, this is the
                                original cosmology when doing cosmology rescaling.
        mag_dataset:       string, name of the dataset of absolute magnitudes to read from
                                the input mock file
    """
    cat = Table.read(input_file)
    
    cat['res'] = np.ones(len(cat['x']), dtype=np.int32)
    cat['box_ind'] = np.arange(len(cat['x']), dtype="i") # this is the index of the galaxy in the original cubic box
    
    # shift coordinates so observer at origin
    cat['x'] -= observer[0]
    cat['y'] -= observer[1]
    cat['z'] -= observer[2]
    
    for dim in 'x','y','z':
        #if np.maximum(np.absolute(cat['z']))
        cat[dim][cat[dim] >  box_size/2.] -= box_size
        cat[dim][cat[dim] <  -box_size/2.] += box_size
    
    print(np.min(cat['x']), np.max(cat['x']))
    print(np.min(cat['y']), np.max(cat['y']))
    print(np.min(cat['z']), np.max(cat['z']))
    
    # get distance corresponding to zmax - needed to find which periodic replications we can skip
    cosmology = hod.cosmo
    rmax = cosmology.comoving_distance(zmax)
    
    print(zmax, rmax)
    
    # loop through periodic replications
    for i in range(-5,6):
        for j in range(-5,6):
            for k in range(-5,6):
                
                #rmin = 0
                #if abs(i) + abs(j) + abs(k) > 0:
                #    rmin = ((abs(i)-0.5)**2 + (abs(j)-0.5)**2 + (abs(k)-0.5)**2)**0.5 * box_size
                
                # calculate rmin, the distance between the observer and the corner of the box replication closest to the observer
                X = np.clip(abs(i)-0.5, 0, 1000) 
                Y = np.clip(abs(j)-0.5, 0, 1000) 
                Z = np.clip(abs(k)-0.5, 0, 1000)
                rmin = (X**2 + Y**2 + Z**2)**0.5 * box_size
                    
                if rmax<rmin: continue # all galaxies in this replication are further than rmax, so we can skip
                
                print('Replication (%i,%i,%i)'%(i,j,k))
    
                cat_new = cut_sky(cat, photsys, hod, Lbox=box_size, replication=(i,j,k), zcut=zmax, mag_cut=mag_faint, z0=snapshot_redshift)
    
                print("Number of galaxies:", np.count_nonzero(cat_new['RA']))

                cat_new.write(output_file[:-5]+'_rep%i%i%i.fits'%(i,j,k), format="fits")


def join_files(galaxy_cutsky, galaxy_cutsky_low, output_file, zmax_low=0.15, zmax=0.6, app_mag_faint=20.2):
    '''
    Combine the cut-sky outputs into a single file
    '''
    
    table = Table()
    
    for file_number in range(34):
        print(file_number)

        table_i = Table()
        table_j = Table()
        
        #loop through the periodic replications
        for i in range(-5,6):
            for j in range(-5,6):
                for k in range(-5,6):
                    
                    try:
                        table_i_rep = Table.read((galaxy_cutsky%file_number)[:-5]+'_rep%i%i%i.fits'%(i,j,k))
                        #print(file_number, i, j, k)
                        table_i = vstack([table_i, table_i_rep])
                        del table_i_rep
                    except:
                        continue
                        
        table_i['FILE_NUM'][:] = file_number
        keep = np.logical_and.reduce((table_i['Z'] >= zmax_low, table_i['Z'] <= zmax, table_i['R_MAG_APP']<=app_mag_faint))
        table_i = table_i[keep]
        
        try:
            table_j = Table.read(galaxy_cutsky_low%file_number)
            table_j['FILE_NUM'][:] = file_number
            keep = np.logical_and(table_j['Z'] < zmax_low, table_j['R_MAG_APP']<=app_mag_faint)
            table_j = table_j[keep]
        except:
            table_j = Table()
        
        table = vstack([table, table_i, table_j])
        del table_j
        
    
    # write the new table
    table.write(output_file, format="fits")
