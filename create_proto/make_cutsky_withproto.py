from hodpy.hod_bgs_abacus import HOD_BGS
from cut_sky_tools import make_lightcone, join_files

cosmo = 0
phase = 0
snapshot_redshift = 0.2
photsys='S'
hod = HOD_BGS(cosmo, photsys=photsys, mag_faint_type='absolute', mag_faint=-10, redshift_evolution=False,
                  replace_central_lookup=True, replace_satellite_lookup=True)

mag_faint_snapshot  = -18 # faintest absolute magitude when populating snapshot
mag_faint_lightcone = -10 # faintest possible absolute magnitude when populating low-z faint lightcone
app_mag_faint = 20.2 # apparent magnitude limit of final cut-sky mock
Lbox = 2000. 
zmax = 0.6 #0.8      # maximum redshift of lightcone
zmax_low = 0.15 # maximum redshift of low-z faint lightcone
mass_cut = 11   # mass cut between unresolved+resolved low z lightcone

#cosmology = CosmologyAbacus(cosmo)
#rmax = cosmology.comoving_distance(zmax)
#rmax_low = cosmology.comoving_distance(zmax_low)

for i in range(1,35):
    part = str(i).zfill(2)
    input_file = 'proto_divided/split_part_%s.fits' % part

    output_file = 'proto_divided/cutskies_flipxy/cutsky_withproto_%s.fits' % part

    make_lightcone(input_file, output_file, hod, photsys, mag_faint=app_mag_faint+0.05, snapshot_redshift=snapshot_redshift, box_size=Lbox, zmax=zmax)

##galaxy_cutsky='proto_divided/cutskies/cutsky_withproto_%s.fits'

##join_files(galaxy_cutsky, output_file='cutsky_with_proto_bgs000_v2.fits', zmax_low=zmax_low, zmax=zmax, app_mag_faint=app_mag_faint)
