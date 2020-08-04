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

import logging
import xarray as xr

logger = logging.getLogger('__name__') #o3asplot

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

    # select period and latitude
    ds_period = ds.sel(time=slice("{}-01-01".format(b_year), 
                                  "{}-12-31".format(e_year)),
                       latitude=slice(lat_max,
                                      lat_min))
    try:
        o3_tco = ds_period[["tco"]]
    except:
        ds_w_tco = ds_period.assign(tco=((ds_period.o3/ds_period.t)
                                         *ds_period.level)*0.724637681159e+19)
        ds_tco = ds_w_tco[["tco"]]

        # Selection phase: calculate tco over column:
        o3_tco = ds_tco.integrate('level')

    logger.debug("o3_tco: {}".format(o3_tco))
    
    return o3_tco.mean(dim=['latitude'])
