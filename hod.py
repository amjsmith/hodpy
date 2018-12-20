#! /usr/bin/env python
from __future__ import print_function
import numpy as np

class HOD(object):

    """
    HOD base class
    """
    
    def number_centrals_mean(self, log_mass, magnitude, redshift):
        raise NotImplementedError

    def number_satellites_mean(self, log_mass, magnitude, redshift):
        raise NotImplementedError

    def number_galaxies_mean(self, log_mass, magnitude, redshift):
        raise NotImplementedError

    def get_number_satellites(self, log_mass, redshift):
        raise NotImplementedError

    def get_magnitude_centrals(self, log_mass, redshift):
        raise NotImplementedError

    def get_magnitude_satellites(self, log_mass, number_satellites, redshift):
        raise NotImplementedError

