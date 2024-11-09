#! /usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator


def get_fcen_array(input_file, snapshots, redshifts):
    '''
    Create a 2D array of the Uchuu fraction of central galaxies, as a function of magnitude
    and redshift. This is done by combining the individual histograms measured from each snapshot
    
    Args:
        input_file: the measured histograms for each snapshot
        snapshots: array of Uchuu snapshot numbers
        redshifts: array of redshifts of each Uchuu snapshot
        
    Returns:
        2D array of central fraction
    '''
    magnitudes = np.arange(-23,-10, 0.5)
    fcen = np.zeros((len(magnitudes),len(redshifts)))

    for i in range(len(snapshots)):
        # read each file
        mag, cen, sat, tot = np.loadtxt(input_file%snapshots[i], delimiter=',', 
                                        skiprows=1, unpack=True)
        f = cen/tot    
    
        # need to do interpolation so that the magnitude bins match
        func = interp1d(mag, f, kind='linear', bounds_error=False, fill_value="extrapolate")
        fcen[:,i] = func(magnitudes)
    return fcen
    
    
def rebin_fcen_array(fcen, redshifts):
    '''
    Rebin the 2D array of central fraction, so the redshifts go 0, 0.1, 0.2, etc
    to make sure 
    
    Args:
        fcen: original 2D array of central fraction
        redshifts: array of redshifts of each Uchuu snapshot
        
    Returns:
        2D array of central fraction after rebinning
    '''
    magnitudes = np.arange(-23,-10, 0.5)
    
    func = RegularGridInterpolator((magnitudes, redshifts), fcen,
                        method='linear', bounds_error=False, fill_value=None)
    
    redshifts_new = np.arange(0,0.81,0.1) # new redshift bins
    fcen_new = np.zeros((len(magnitudes),len(redshifts_new)))
    
    for i in range(len(redshifts_new)):
        fcen_new[:,i] = func((magnitudes, np.ones(len(magnitudes))*redshifts_new[i]))
    
    return fcen_new
    


if __name__ == '__main__':
    
    # uchuu snapshot numbers and corresponding redshifts
    snapshots = 40, 41, 43, 45, 47, 50
    redshifts = 0.49, 0.43, 0.30, 0.19, 0.093, 0
    
    input_file = 'histogram_%i.csv'
    output_file = 'central_fraction_uchuu.npy'
    
    # create a 2D array of central fraction
    fcen = get_fcen_array(input_file, snapshots, redshifts)
    
    # rebin the 2D array so the redshifts match what is expected
    fcen = rebin_fcen_array(fcen, redshifts)
        
    # save the lookup files
    np.save(output_file, fcen)
