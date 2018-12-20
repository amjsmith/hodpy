# hod

This repository contains code for applying the HOD method described in Smith et al. 2017 (https://arxiv.org/abs/1701.06581).
This code can be used to populate either a halo lightcone, or a halo catalogue from a simulation snapshot.

For testing purposes, a small section of the MXXL halo lightcone is provided in `input/halo_catalogue_small.hdf5`. This halo
catalogue covers an area of 25 sq deg (0<ra<5 and 0<dec<5), to z=1. The full sky halo catalogue to z=2.2 is available at

http://icc.dur.ac.uk/data/

https://tao.asvo.org.au/tao/

A small section of the MXXL box at snapshot 58 (corresponding to z=0.116) is also provided in `input/snapshot_58_small.hdf5`.
The snapshot has been cut to only the most massive subhaloes in each FOF group, in a box that is 300 Mpc/h on each side 
(0.1% of the volume of the full MXXL box).

# Populating a lightcone

## make_catalogue.py

Contains the main program for populating a halo lightcone with galaxies. 
Running this will read in the small input halo catalogue, `input/halo_cataloge_small.hdf5`, and populate it 
with galaxies using the
HOD prescription of Smith+17, assigning each galaxy an r-band magnitude, and g-r colour. The output galaxy 
catalogue will be stored in the file output/cat.hdf5

### Input file properties

Properties stored in `input/halo_cataloge_small.hdf5` are described below. Not all of these properties are used in the code.
Properties that are needed are indicated in bold, while properties that are not used are in italics.

- *M200c*: M200crit, the halo mass in spheres with average density 200 times the critical density, in units 1e10 Msun/h

- **M200m**: M200mean, the halo mass in spheres with average density 200 times the mean density, in units 1e10 Msun/h

- **dec**: Declination, in degrees

- **ra**: Right ascension, in degrees

- **rvmax**: Radius at which Vmax occurs, in (comoving) Mpc/h

- *vdisp*: Velocity dispersion, in (proper) km/s

- *vmax*: Vmax, the maximum circular velocity, in (proper) km/s

- **z_cos**: Cosmological redshift, which ignores the peculiar velocity

- **z_obs**: Observed redshift, which takes into account the peculiar velocity

### Output file properties
 
Running make_catalogue.py will populate `input/halo_cataloge_small.hdf5`, and store the galaxy catalogue in `output/cat.hdf5`.
The properties in the galaxy catalogue are described here.

- **abs_mag**: r-band absolute magnitude, k-corrected to z=0.1, with h=1

- **app_mag**: r-band apparent magnitude

- **cen_ind**: Index of the central galaxy in the same halo as this galaxy

- **col**: g-r colour, k-corrected to z=0.1

- **dec**: Declination, in degrees

- **halo_ind**: Index of halo in the input catalogue

- **halo_mass**: Mass of the host halo (defined as M200mean), in Msun/h

- **is_cen**: True if the galaxy is a central, False if the galaxy is a satellite

- **ra**: Right ascension, in degrees

- **zcos**: Cosmological redshift, ignoring velocities

- **zobs**: Observed redshift, taking into account the peculiar velocity along the line of sight

The exact galaxy properties to store, and additional properties from the halo catalogue, can be specified in the 
save_to_file method (see `galaxy_catalogue.py`). By default, all galaxy properties, and no halo properties, will be stored.

### Running the code

The code works with either Python 2 or Python 3, and can simply be run with
```
python make_catalogue.py
```
Note that the very first time the code is run, the class `HOD_BGS` in `hod.py` constructs a large lookup table of central
and satellite galaxy luminosities, which takes some time. This is done in order to speed up the root finding 
procedure used to assign magnitudes to each galaxy. These tables are written to the files 
`lookup/central_magnitudes.npy` and `lookup/satellite_magnitudes.npy`. 
The next time the code is run, the lookup tables are read from these files, making the code much faster.

If any changes are made to the input HODs, these files must be deleted. The lookup tables will then be recaluated
for the new HODs the next time the code is run.

# Populating a snapshot

## make_catalogue_snapshot.py

Contains the main program for populating a simulation snapshot with galaxies. 
Running this will read in a small section of one of the MXXL snapshots, `input/snapshot_58_small.hdf5`, 
and populate it with galaxies using the
HOD prescription of Smith+17, assigning each galaxy an r-band magnitude, and g-r colour. The output galaxy 
catalogue will be stored in the file `output/cat.hdf5`

### Input file properties

The properties stored in `input/snapshot_58_small.hdf5` are described below. Not all of these properties are used 
in the code.
Properties that are needed are indicated in bold, while properties that are not used are in italics.

- **M200m**: M200mean, the halo mass in spheres with average density 200 times the mean density, in units 1e10 Msun/h

- **pos**: 3D Cartesian position vector, in (comoving) Mpc/h

- **rvmax**: Radius at which Vmax occurs, in (comoving) Mpc/h

- **vel**: 3D Cartesian velocity vector, in (proper) km/s

- *vmax*: Vmax, the maximum circular velocity, in (proper) km/s

### Output file properties

Running make_catalogue.py will populate `input/snapshot_58_small.hdf5`, and store the galaxy 
catalogue in `output/cat.hdf5`.
The properties in the galaxy catalogue are described here.

- **abs_mag**: r-band absolute magnitude, k-corrected to z=0.1, with h=1

- **cen_ind**: Index of the central galaxy in the same halo as this galaxy

- **col**: g-r colour, k-corrected to z=0.1

- **halo_ind**: Index of halo in the input catalogue

- **halo_mass**: Mass of the host halo (defined as M200mean), in Msun/h

- **is_cen**: True if the galaxy is a central, False if the galaxy is a satellite

- **pos**: 3D Cartesian position vector, in (comoving) Mpc/h

- **vel**: 3D Cartesian velocity vector, in (proper) km/s

- **zcos**: Cosmological redshift (the redshift of the simulation snapshot)

As with the lightcone, the exact galaxy properties to store, and additional properties from the halo catalogue, 
can be specified in the save_to_file method (see `galaxy_catalogue_snapshot.py`). 
By default, all galaxy properties, and no halo properties, will be stored.

### Running the code

Before running the code, the input catalogue in `parameters.py` must first be set to 
`halo_file = "input/snapshot_58_small.hdf5"`. The code can then simply be run with
```
python make_catalogue_snapshot.py
```
As with the lightcone code, the first time it is run, the code will generate large lookup tables of galaxy
luminosities. 


# Parameters

## parameters.py

This is where various parameters in the code can be set. These parameters are described here. Input files indicated
in brackets will be created if they don't exist when running the code.

- `lookup_dir`: Directory in which various lookup files are located (see the next section for details on these files)

- `halo_file`: Location of the input halo lightcone or snapshot file

- `snapshot`: Snapshot number. This is only required when populating a snapshot. The class `MXXL_Snapshot` in 
`halo_catalogue.py` contains a method for converting the MXXL snapshot number to the corresponding redshift.

- `mag_faint`: Faint apparent magnitude limit. When populating a lightcone, the catalogue is cut to this limit
at the very end.

- (`lookup_central`): Location of lookup table of central galaxy luminosities. If this file already exists, it will be
read from. If it doesn't exist, this file will be created.

- (`lookup_satellite`): Location of lookup table of satellite galaxy luminosities. If this file already exists, it will be
read from. If it doesn't exist, this file will be created.

- `lookup_snapshots`: Location of lookup table of simulation snapshots and the corresponding redshift. This is only needed
when populating a snapshot.

- `box_size`: Size of the simulation cubic box, in Mpc/h. This is only needed when populating a snapshot.

#### Cosmology

The code assumes a flat LCDM cosmology

- `h0`: Present day Hubble parameter, in units of 100 km/s/Mpc

- `OmegaM`: Mass density parameter

- `OmegaL`: Lambda density parameter

#### Target luminosity function

The following parameters set the shape of the target luminosity function. At low z (z<0.1), the shape is set by the file `lf_file`, while at higher z (z>0.2), it is set by a Schecter function, with a smooth transition between 0.1<z<0.2. 

- (`lf_file`): Tabulated file of the cumulative target luminoisity function at low redshifts. This is the luminosity function at
z=0.1 predicted from the HODs (i.e. the result of integrating over all masses the halo mass function multiplied by the HOD).
This transitions to the Blanton SDSS luminosity function at the faint end, and is extrapolated using a power law.
If this file is not provided, the target luminosity function will be calculated and saved to this file.

- `lf_sdss`: Tabulated file of the Blanton SDSS cumulative luminosity function.

- `Phi_star`: Schechter function normalization (h^3/Mpc^3)

- `M_star`: Schechter function characteristic absoulte magnitude (M - 5logh)

- `alpha`: Schechter function faint end slope

- `P`:  Number density evolution parameter

- `Q`: Magnitude evolution parameter

- `k_corr_file`: File containing the polynomial coefficients of the colour-dependent k-corrections (see Smith+17)

#### Mass function parameters

- `mf_fits_file`: File containing the fits to the simulation mass function, with the same form as the Sheth-Tormen
mass function, at each simulation snapshot

- `deltacrit_file`: Tabulated file of delta_c(z), in the cosmology of the simulation.

- `sigma_file`: Tabulated file of sigma(M), in the cosmology of the simulation.

#### HOD parameters

- (`slide_file`): File containing the parameters used to evolve the HODs with redshift and reproduce the target luminosity
function


# Other files

## halo_catalogue.py

## galaxy_catalogue.py

## hod.py

## luminosity_function.py

## colour.py

## k_correction.py

##
