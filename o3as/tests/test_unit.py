# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat June 30 23:47:51 2020
@author: vykozlov
"""
import numpy as np
import os
import pandas as pd
import pkg_resources
import xarray as xr
import unittest
from o3as import plothelpers as phlp

class TestModelMethods(unittest.TestCase):

    def setUp(self):
        module = __name__.split('.', 1)
        pkg = pkg_resources.get_distribution(module[0])
        self.meta = {
            'name' : None,
            'version' : None,
            'summary' : None,
            'home-page' : None,
            'author' : None,
            'author-email' : None,
            'license' : None
        }
        for line in pkg.get_metadata_lines("PKG-INFO"):
            line_low = line.lower() # to avoid inconsistency due to letter cases
            for par in self.meta:
                if line_low.startswith(par.lower() + ":"):
                    _, value = line.split(": ", 1)
                    self.meta[par] = value

        # create artificial data and store it
        delta_years = 2
        self.start_date = (np.datetime64('today', 'M') - 
                           np.timedelta64(12*delta_years, 'M'))
        self.end_date = (self.start_date + 
                         np.timedelta64(12*delta_years - 1, 'M'))
        self.o3ds = xr.Dataset(
            {"t": (("level", "latitude", "time"), np.ones((10, 19, 24))),
             "o3": (("level", "latitude", "time"), np.ones((10, 19, 24))),
            },
            coords={
                    "level": [z for z in range(0, 1000, 100)],
                    "latitude": [x for x in range(-90, 100, 10)],
                    "time": [ self.start_date + np.timedelta64(x, 'M') 
                              for x in range(0, 12*delta_years, 1)]
                   }
        )

        end_year = np.datetime64('today', 'Y').astype(int) + 1970
        begin_year = end_year - delta_years

        self.data_base_path = "tmp/data"
        os.environ["O3AS_DATA_BASE_PATH"] = self.data_base_path
        model = "o3as-test"
        test_dir = os.path.join(self.data_base_path, model) 
        self.pattern=os.path.join(test_dir, "*_skim-*.nc")
        test_path  = os.path.join(test_dir, model + "_skim-" + 
                                            str(end_year) + ".nc")
        os.makedirs(test_dir, exist_ok=True)
        self.o3ds.to_netcdf(test_path)
        #self.o3ds.close()

        self.kwargs = {
            'type': 'tco',
            'models': [model],
            'begin_year': begin_year,
            'end_year': end_year,
            'lat_min': -10,
            'lat_max': 10
        }

        print(self.kwargs)

    def test_metadata_type(self):
        """
        Test that self.meta is dict
        """
        self.assertTrue(type(self.meta) is dict)


    def test_metadata_values(self):
        """
        Test that metadata contains right values (subset)
        """
        emails = "tobias.kerzenmacher@kit.edu,\
        borja.sanchis@kit.edu, valentin.kozlov@kit.edu"
        self.assertEqual(self.meta['name'].replace('-','_'),
                        'o3as'.replace('-','_'))
        self.assertEqual(self.meta['author'], 'KIT-IMK')
        self.assertEqual(self.meta['author-email'].lower().replace(' ',''), 
                         emails.lower().replace(' ',''))
        self.assertEqual(self.meta['license'], 'GNU LGPLv3')


    def test_get_datafiles(self):
        """
        Test if one gets the files pattern,
        by checking that the directory in the pattern exists
        """
        model = self.kwargs['models'][0]
        test_pattern = phlp.get_datafiles(model)
        print("[test_pattern]: {}".format(test_pattern))
        dir_name = os.path.dirname(test_pattern)
        self.assertTrue(os.path.isdir(dir_name))

        
    def test_get_dataset_type(self):
        """
        Test that the returned dataset type is correct, xarray.Dataset
        """
        print(self.pattern)
        ds = phlp.get_dataset(self.pattern)
        self.assertTrue(type(ds) is xr.Dataset)


    def test_get_dataset_values(self):
        """
        Test that returned dataset values are the same as generated.
        """
        print(self.pattern)
        ds = phlp.get_dataset(self.pattern)
        self.assertEqual(ds, self.o3ds)

        
    def test_get_date_range(self):
        """
        Test correctness of returned min/max dates
        """
        date_min, date_max = phlp.get_date_range(self.o3ds)
        self.assertEqual(date_min, self.start_date)
        self.assertEqual(date_max, self.end_date)


    def test_get_periodicity(self):
        """
        Test correctness of returned periodicity
        """
        time_axis = pd.DatetimeIndex(self.o3ds.coords['time'].values)
        period = phlp.get_periodicity(time_axis)
        self.assertEqual(period, 12)

if __name__ == '__main__':
    unittest.main()
