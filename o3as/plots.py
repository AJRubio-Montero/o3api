# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Mon Aug  3 15:48:04 2020

@author: vykozlov
"""

import numpy as np
import o3as.config as cfg
import os
import logging
import xarray as xr

logger = logging.getLogger('__name__') #o3asplot
logger.setLevel(cfg.log_level)


class DataSelection:
    """Class describing the basic data selection"""

    def __init__ (self, ptype, **kwargs):
        self.ptype = ptype
        self.model = 'dummy'
        self.b_year = kwargs['begin_year']
        self.e_year = kwargs['end_year']
        self.lat_min = kwargs['lat_min']
        self.lat_max = kwargs['lat_max']
        self.data_pattern = self.ptype + "*_????.nc"
        self.datafiles = os.path.join(cfg.O3AS_DATA_BASEPATH, 
                                       self.model, 
                                       self.data_pattern)

    def __set_datafiles(self, model):
        self.model = model
        self.datafiles = os.path.join(cfg.O3AS_DATA_BASEPATH, 
                                      self.model, 
                                      self.data_pattern)


    def __get_dataset(self):
        """Load data from the file list
        :return: xarray dataset
        """
        # Check: http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
        # chunks={'latitude': 8} - very machine dependent!
        # laptop (RAM 8GB) : 8, lsdf-gpu (128GB) : 64
        # engine='h5netcdf' : need h5netcdf files? yes, but didn't see improve
        chunk_size = int(os.getenv('O3AS_CHUNK_SIZE', -1))
        logger.debug("Chunk Size: {}".format(chunk_size))

        if chunk_size > 0:
            ds = xr.open_mfdataset(self.datafiles, chunks={'lat': chunk_size },
                                   concat_dim='time',
                                   data_vars='minimal', coords='minimal',
                                   parallel=True)
        else:
            ds = xr.open_mfdataset(self.datafiles,
                                   concat_dim='time',
                                   data_vars='minimal', coords='minimal',
                                   parallel=True)
    
        return ds


    def __check_latitude_order(self, ds):
        """Function to check latitude order
        :param ds: xarray dataset to check
        :return: lat_0, lat_last
        """
        lat_0 = np.amin(ds.coords['lat'].values[0]) # min latitude
        lat_last = np.amax(ds.coords['lat'].values[-1]) # max latitude

        return lat_0, lat_last

        
    def get_dataslice(self, model):
        """Function to select slice of data"""

        self.__set_datafiles(model)
        ds = self.__get_dataset()
        logger.info("Dataset is loaded from storage location: {}".format(ds))
        # check in what order latitude is used, e.g. (-90..90) or (90..-90)
        lat_0, lat_last = self.__check_latitude_order(ds)
        logger.debug("ds: lat_0 = {}, lat_last: {}".format(lat_0, lat_last))
        if lat_0 < lat_last:
            lat_a = self.lat_min
            lat_b = self.lat_max
        else:
            lat_a = self.lat_max
            lat_b = self.lat_min

        # select data according to the period and latitude
        # BUG(?) ccmi-umukca-ucam complains about 31-12-year, but 30-12-year works
        ds_slice = ds.sel(time=slice("{}-01-01T00:00:00".format(self.b_year), 
                                      "{}-12-30T23:59:59".format(self.e_year)),
                          lat=slice(lat_a,
                                    lat_b))  # latitude
                                     
        return ds_slice


class ProcessForTCO3(DataSelection):
    """Subclass to calculate TCO3"""
    def __init__(self, **kwargs):
        super().__init__('tco3_zm', **kwargs)

    def get_plot_data(self, model):
        """Return plot data"""
        # data selection according to time and latitude
        ds_slice = super().get_dataslice(model)
        ds_tco3 = ds_slice[["tco3_zm"]]
        logger.debug("ds_tco3: {}".format(ds_tco3))

        return ds_tco3.mean(dim=['lat'])


class ProcessForVMRO3(DataSelection):
    """Subclass to calculate VMRO3"""
    def __init__(self, **kwargs):
        super().__init__('vmro3_zm', **kwargs)

    def get_plot_data(self, model):
        """Return plot data"""
        # data selection according to time and latitude
        ds_slice = super().get_dataslice(model)
        ds_vmro3 = ds_slice[["vmro3_zm"]]
        logger.debug("ds_vmro3: {}".format(ds_vmro3))

        return ds_vmro3.mean(dim=['lat'])
        

class ProcessForTCO3Return(DataSelection):
    """Subclass to calculate TCO3_Return"""
    def __init__(self, **kwargs):
        super().__init__('tco3_return', **kwargs)

    def get_plot_data(self, model):
        """Return plot data"""
        # data selection according to time and latitude
        ds_slice = super().get_dataslice(model)
        ds_tco3_return = ds_slice[["tco3_return"]]
        logger.debug("ds_tco3_return: {}".format(ds_tco3_return))

        return ds_tco3_return.mean(dim=['lat'])
