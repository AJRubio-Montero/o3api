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

import o3as.config as cfg
import logging
import numpy as np

logger = logging.getLogger('__name__') #o3asplot
logger.setLevel(cfg.log_level)

def check_latitude_order(ds):
    """
    Function to check latitude order
    :param ds: xarray dataset to check
    :return: lat_0, lat_last
    """
    lat_0 = np.amin(ds.coords['lat'].values[0]) # latitude
    lat_last = np.amax(ds.coords['lat'].values[-1]) # latitude

    return lat_0, lat_last

def process_for_tco(**kwargs):
    """Data processing for TCO plot
    :param kwargs: provided in the API call parameters, expected: 
         ds: xarray dataset
         begin_year: year to begin data processing from
         end_year: year to end data processing
         lat_min: minimal latitude for data selection
         lat_max: maximal latitude for data selection
    :return: xarray dataset with the calculated tco parameter
    """
    ds = kwargs['ds']
    b_year = kwargs['begin_year']
    e_year = kwargs['end_year']
    lat_min = kwargs['lat_min']
    lat_max = kwargs['lat_max']

    # check in what order latitude is used, e.g. (-90..90) or (90..-90)
    lat_0, lat_last = check_latitude_order(ds)
    logger.debug("ds: lat_0 = {}, lat_last: {}".format(lat_0, lat_last))
    if lat_0 < lat_last:
        lat_a = lat_min
        lat_b = lat_max
    else:
        lat_a = lat_max
        lat_b = lat_min
    # select period and latitude
    # BUG(?) ccmi-umukca-ucam complains about 31-12-year, but 30-12-year works
    ds_period = ds.sel(time=slice("{}-01-01T00:00:00".format(b_year), 
                                  "{}-12-30T23:59:59".format(e_year)),
                       lat=slice(lat_a,
                                 lat_b))  # latitude
    try:
        o3_tco = ds_period[["tco3_zm"]]
    except:
        ds_w_tco = ds_period.assign(tco=((ds_period.o3/ds_period.t)
                                         *ds_period.level)*1.45e+6)
                                         # 0.724637681159e+19
        ds_tco = ds_w_tco[["tco3_zm"]]

        # Selection phase: calculate tco over column:
        o3_tco = ds_tco.integrate('level')

    logger.debug("o3_tco: {}".format(o3_tco))
    
    return o3_tco.mean(dim=['lat'])
