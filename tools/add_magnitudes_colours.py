import numpy as np
import fitsio
from astropy.table import Table
import sys
sys.path.append('..')

from hodpy.cosmology import CosmologyDESI
from hodpy.k_correction import DESI_KCorrection, DESI_KCorrection_color 

def add_magnitudes_colours(filename):
    """
    Add rest-frame g-r colours and absolute magnitudes to the BGS
    data clustering catalogues

    Args:
        filename: filename of the clustering catalogue
    """
    print("Adding ABSMAG_RP1 and REST_GMR_0P1 to", filename)
    
    cosmo = CosmologyDESI()
    
    # read file and convert fluxes to apparent magnitudes
    data = Table.read(filename)

    try:
        # try to read flux dered from catalogue
        flux_r_dered = data['flux_r_dered']
        flux_g_dered = data['flux_g_dered']
        z = data['Z']
    except:
        # calcuate it if column doesn't exist
        flux_r_dered = data['FLUX_R']/data['MW_TRANSMISSION_R']
        flux_g_dered = data['FLUX_G']/data['MW_TRANSMISSION_G']
        z = data['Z_not4clus']

    # convert fluxes to apparent magnitudes
    D_L = cosmo.comoving_distance(z) * (1+z)
    distmod = 5*np.log10(D_L) + 25
        
    rmag_dered = 22.5 - 2.5*np.log10(np.maximum(1.0e-10,flux_r_dered))  #Note not deredening correction as fluxes in Fastspec catalogue are already dereddened
    gmag_dered = 22.5 - 2.5*np.log10(np.maximum(10.e-10,flux_g_dered))
    obs_gmr = gmag_dered - rmag_dered
    
    # apply k-corrections to get rest-frame colours and absolute magnitudes
    # these are different in North and South
    rest_gmr = np.zeros(len(obs_gmr))
    absmag_r = np.zeros(len(rmag_dered))

    for photsys in 'N','S':    
        in_reg = data['PHOTSYS'] == photsys
        
        # convert observed colour to rest-frame
        kcorr_col = DESI_KCorrection_color(photsys=photsys)

        z_reg = np.clip(z[in_reg],-0.5,1.0)
        is_nan = np.isnan(z_reg)
        z_reg[is_nan] = 0

        obs_gmr_reg = np.clip(obs_gmr[in_reg],-3,3)
        is_nan = np.isnan(obs_gmr_reg)
        obs_gmr_reg[is_nan] = 0
    
        rest_gmr[in_reg] = kcorr_col.rest_frame_colour(z_reg, obs_gmr_reg)
                                                       
        
        # convert apparent to absolute magnitude
        kcorr_r = DESI_KCorrection(band='r', photsys=photsys, cosmology=cosmo)
        absmag_r[in_reg] = kcorr_r.absolute_magnitude(rmag_dered[in_reg], 
            z[in_reg], rest_gmr[in_reg], use_ecorr=True, Q=0.67, zq=0.1)

    data['REST_GMR_0P1'] = rest_gmr
    data['ABSMAG_R'] = absmag_r
    data['ABSMAG_RP1'] = absmag_r
    
    data.write(filename, overwrite=True)
          
          
if __name__ == '__main__':
    
    filename = sys.argv[1]
    
    #path = '/pscratch/sd/a/amjsmith/desi/main/LSS/iron/LSScats/test/'
    #filename = 'BGS_BRIGHT_clustering.dat.fits'
    
    add_magnitudes_colours(filename)
