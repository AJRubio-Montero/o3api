# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
"""
Created on Wed Aug  5 09:53:40 2020

@author: vykozlov
"""

import o3as.config as cfg
import logging
import numpy as np
import o3as.plots as o3plots
import os
import xarray as xr

# conigure python logger
logger = logging.getLogger('__name__') #o3asplot
logger.setLevel(cfg.log_level)

pconf = cfg.plot_conf

def get_datafiles(model):
    """Return pattern for files corresponding to the model
    :param model: model name, also used to define path where to look for files,
          e.g. as O3AS_DATA_BASE_PATH/model
    :type model: string
    :return: pattern for files
    """
    # where to look for files. 
    # Default is /srv/o3as/data/ + model but
    # one can change to $DATA_BASE_PATH + model
    data_base_path = os.getenv('O3AS_DATA_BASE_PATH', "/srv/o3as/data/")
    data_path = os.path.join(data_base_path, model)

    return os.path.join(data_path,"*_skim-*.nc")


def get_dataset(files):
    """Load data from the file list
    :param files: list of files or file pattern with data
    :return: xarray dataset
    """
    # Check: http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    # chunks={'latitude': 8} - very machine dependent!
    # laptop (RAM 8GB) : 8, lsdf-gpu (128GB) : 64
    # engine='h5netcdf' : need h5netcdf files? yes, but didn't see improve
    chunk_size = int(os.getenv('O3AS_CHUNK_SIZE', -1))
    logger.debug("Chunk Size: {}".format(chunk_size))

    if chunk_size > 0:
        ds = xr.open_mfdataset(files, chunks={'latitude': chunk_size },
                               concat_dim=pconf['time_c'],
                               data_vars='minimal', coords='minimal',
                               parallel=True)
    else:
        ds = xr.open_mfdataset(files,
                               concat_dim=pconf['time_c'],
                               data_vars='minimal', coords='minimal',
                               parallel=True)

    logger.info("Dataset is loaded from storage location: {}".format(ds))
    
    return ds


def get_date_range(ds):
    """Return range of dates in the provided data
    :param ds: xarray dataset to check
    :return: date_min, date_max
    """
    date_min = np.amin(ds.coords[pconf['time_c']].values)
    date_max = np.amax(ds.coords[pconf['time_c']].values)

    return date_min, date_max


def get_periodicity(pd_time):
    """Calculate periodicity in the provided data
    :param pd_time: pandas DatetimeIndex
    :return: calculated periodicity as number of points per year
    :rtype: int
    """
    date_range = np.amax(pd_time) - np.amin(pd_time)
    date_range = (date_range/np.timedelta64(1, 'D'))
    periodicity = ((pd_time.size - 1) / date_range ) * 365.0
    logger.debug("Periodicity calculated: {}".format(periodicity))

    return int(round(periodicity, 0))


def set_plot_title(**kwargs):
    """Set plot title
    :param kwargs: provided in the API call parameters
    :return: plot_title with added input parameters
    :rtype: string
    """
    plot_type = kwargs[pconf['plot_t']]
    plot_title = plot_type + " (inputs: "
    for par in pconf[plot_type]['inputs']:
        plot_title += str(kwargs[par]) + ","

    plot_title = plot_title[:-1] + ")" # replace last "," with ")"
    
    return plot_title

    
def set_file_name(**kwargs):
    """Set file name
    :param kwargs: provided in the API call parameters
    :return: file_name with added input parameters (no extension given!)
    :rtype: string
    """
    plot_type = kwargs[pconf['plot_t']]
    file_name = plot_type
    for par in pconf[plot_type]['inputs']:
        file_name += "_" + str(kwargs[par])

    return file_name


def process(**kwargs):
    """Select data processing according to the plot type
    :param kwargs: provided in the API call parameters
    :return: processed data
    """

    plot_type = kwargs[pconf['plot_t']]
    data = kwargs['ds']
    
    if plot_type == 'tco':
        data_processed = o3plots.process_for_tco(**kwargs)
    elif plot_type == 'xx':
        data_processed = data
    else:
        # should return something
        data_processed = data

    return data_processed
