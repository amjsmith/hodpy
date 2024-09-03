#! /usr/bin/env python
import numpy as np

def is_negative_zero(array):
    """
    Checks the values of a numpy array for negative zero values (-0.0)

    Args:
        array: numpy array of floats or integers
    Returns:
        boolean array, True for elements equal to -0.0, False otherwise
    """
    return np.logical_and(array == 0, np.copysign(1, array) == -1)


class Catalogue(object):
    """
    Catalogue of objects on the sky

    Args:
        cosmology: object of the class Cosmology
    """
    def __init__(self, cosmology):
        self._quantities = {}
        self.size = 0
        self.cosmology = cosmology


    def get(self, prop):
        """
        Get property from catalogue

        Args:
            prop: string of the name of the property
        Returns:
            array of property
        """
        return self._quantities[prop]


    def add(self, prop, value):
        """
        Add property to catalogue

        Args:
            prop:  string of the name of the property
            value: array of values of property
        """
        self._quantities[prop] = value


    def cut(self, keep):
        """
        Cut catalogue to mask

        Args:
            keep: boolean array
        """
        for quantity in self._quantities:
            self._quantities[quantity] = self._quantities[quantity][keep]
        self.size = np.count_nonzero(keep)


    def equatorial_to_pos3d(self, ra, dec, z):
        """
        Convert ra, dec, z to 3d cartesian coordinates

        Args:
            ra:  array of ra [deg]
            dec: array of dec [deg]
            z:   array of redshift
        Returns:
            2d array of position vectors [Mpc/h]
        """
        # convert degrees to radians
        ra *= np.pi / 180
        dec *= np.pi / 180

        # comoving distance to redshift z
        r_com = self.cosmology.comoving_distance(z)

        pos = np.zeros((len(r_com),3))
        pos[:,0] = r_com * np.cos(ra) * np.cos(dec) # x coord
        pos[:,1] = r_com * np.sin(ra) * np.cos(dec) # y coord
        pos[:,2] = r_com * np.sin(dec)              # z coord

        return pos


    def pos3d_to_equatorial(self, pos):
        """
        Convert 3d cartesian coordinates to ra, dec, z

        Args:
            pos: 2d array of position vectors [Mpc/h]
        Returns:
            ra:  array of ra [deg]
            dec: array of dec [deg]
            z:   array of redshift
        """

        # get ra
        ra = np.arctan(pos[:,1] / pos[:,0])
        ind = np.logical_and(pos[:,1] < 0, pos[:,0] >= 0)
        ra[ind] += 2*np.pi
        ind = pos[:,0] < 0
        ra[ind] += np.pi

        # Fix some rare edge cases where x=0
        ind = is_negative_zero(pos[:,0])
        ra[ind] += np.pi
        ind = np.logical_and(is_negative_zero(pos[:,0]), pos[:,1]<0)
        ra[ind] -= 2*np.pi
    
        # If x=0 and y=0, set RA=0 to avoid NaN
        ind = np.logical_and(pos[:,0]==0, pos[:,1]==0)
        ra[ind] = 0
        
        # get z from comoving distance
        r_com = np.sqrt(np.sum(pos**2, axis=1))
        z = self.cosmology.redshift(r_com)
        
        # get dec
        dec = (np.pi/2) - np.arccos(pos[:,2] / r_com)

        # convert radians to degrees
        ra *= 180 / np.pi
        dec *= 180 / np.pi

        return ra, dec, z


    def vel_to_zobs(self, z_cos, v_los):
        """
        Convert line of sight velocity to observed redshift

        Args:
            z_cos: array of cosmological redshift
            v_los: array of line of sight velocity [km/s]
        Returns:
            array of observed redshift
        """
        z_obs = ((1 + z_cos) * (1 + v_los/3e5)) - 1.
        return z_obs


    def zobs_to_vel(self, z_cos, z_obs):
        """
        Convert observed redshift to line of sight velocity

        Args:
            z_cos: array of cosmological redshift
            z_obs: array of observed redshift
        Returns:
            array of line of sight velocity [km/s]
        """
        v_los = 3e5 * ((1. + z_obs)/(1. + z_cos) - 1)
        return v_los


    def vel_to_vlos(self, pos, vel):
        """
        Projects velocity vector along line of sight, with observer positioned at the origin

        Args:
            pos: 2d array of comoving position vectors [Mpc/h]
            vel: 2d array of velocity vectors [km/s]
        Returns:
            array of line of sight velocity [km/s]
        """
        # comoving distance to each object
        distance = np.sum(pos**2, axis=1)**0.5

        # normalize postion vectors
        pos_norm = pos.copy()
        for i in range(3):
            pos_norm[:,i] = pos_norm[:,i] / distance

        # project velocity along position vectors
        v_los = np.sum(pos_norm*vel,axis=1)

        return v_los



    def save_to_file(self, file_name, format, properties=None):
        """
        Save catalogue to file. The properties to store can be specified
        using the properties argument. If no properties are specified,
        the full catalogue will be saved.

        Args:
            file_name: string of file_name
            format:    string of file format
            properties: (optional) list of properties to save
        """

        directory = '/'.join(file_name.split('/')[:-1])
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        if format == "hdf5":
            import h5py

            f = h5py.File(file_name, "a")

            if properties is None: 
                # save every property
                for quantity in self._quantities:
                    f.create_dataset(quantity, data=self._quantities[quantity],
                                     compression="gzip")
            else: 
                # save specified properties
                for quantity in properties:
                    f.create_dataset(quantity, data=self._quantities[quantity],
                                     compression="gzip")
            f.close()

        elif format == "fits":
            from astropy.table import Table
            
            if properties is None:
                # save every property
                t = Table(list(self._quantities.values()), 
                          names=list(self._quantities.keys()))
                t.write(file_name, format="fits")
            else:
                # save specified properties
                data = [None] * len(properties)
                for i, prop in enumerate(properties):
                    data[i] = self._quantities[prop]
                t = Table(data, names=properties)
                t.write(file_name, format="fits")

        # can add more file formats...

        else:
            raise ValueError("Invalid file format")
