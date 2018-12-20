# Parameters

# directory in which lookup files are stored
lookup_dir = "lookup/"

# input halo catalogue
halo_file = "input/halo_catalogue_small.hdf5"   # lightcone
#halo_file = "input/snapshot_58_small.hdf5"       # snapshot
snapshot = 58                                    # snapshot

# output directory
output_dir = "output/"

# apparent magnitude threshold
mag_faint = 20.0

# lookup tables of central and satellites magnitudes
# these files will be created if they don't exist
lookup_central = lookup_dir+"central_magnitudes.npy"
lookup_satellite = lookup_dir+"satellite_magnitudes.npy"

### Simulation ### (Only needed if populating a snapshot)
lookup_snapshots = lookup_dir+"mxxl_snapshots.dat"
box_size = 3000 # Mpc/h

### Cosmology ###
h0     = 0.73
OmegaM = 0.25
OmegaL = 0.75
 
### Luminosity function ###
lf_file  = lookup_dir+"target_lf.dat" # cumulative target LF at z=0.1
lf_sdss  = lookup_dir+"sdss_cumulative_lf.dat" # sdss cumulative LF
Phi_star = 0.94e-2 # Schecheter params at high z
M_star   = -20.70
alpha    = -1.23
P        = 1.8     # evolution params
Q        = 0.7

# k-correction
k_corr_file = lookup_dir+"k_corr_rband_z01.dat"

### Mass function ###
mf_fits_file   = lookup_dir+"mf_fits.dat"   # fit to MF of simulation
deltacrit_file = lookup_dir+"deltacrit.dat" # delta_crit(z)
sigma_file     = lookup_dir+"sigma.dat"     # sigma(M)

### HOD parameters ###
slide_file = lookup_dir+"slide_factors.dat"
# Mmin
Mmin_Ls = 3.91841775e+09
Mmin_Mt = 3.06648452e+11
Mmin_am = 2.57628181e-01
# M1
M1_Ls = 3.70558341e+09
M1_Mt = 4.77678420e+12
M1_am = 3.05963831e-01
# M0
M0_A = 1.78353177
M0_B = -5.98024172
# alpha
alpha_A = 0.0982841
alpha_B = 80.27598
alpha_C = 10.0
# sigma_logM
sigma_A = 0.02583643
sigma_B = 0.68126852
sigma_C = 21.05
sigma_D = 2.5
