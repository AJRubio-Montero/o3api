#!/usr/bin/env python3
#
# Script to process selected data and 
# return either PDF plot or JSON document.
# Used to build REST API.
#
## Ozone related information: ##
# latitude: index for geolocation
# time: index for time (e.g. hours since start time - 6 hourly spacing)
# level: index for pressure / altitude (e.g. hPa)
# t: temperature
# o3: ozone data
# tco: total column ozone

# ToDo: improve Error handling, that Errors are correctly returned by API
#       e.g. raise OSError("no files to open")

import logging
import matplotlib.pyplot as plt
import numpy as np
import o3as.plots as o3plots
import os
import pandas as pd
import time
import xarray as xr

from flask import send_file
from flask import jsonify, make_response, request
from fpdf import FPDF
from functools import wraps
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose # accurate enough

# conigure python logger
logger = logging.getLogger('__name__') #o3asplot
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s')
logger.setLevel(logging.DEBUG)

## Authorization
from flaat import Flaat
flaat = Flaat()

# list of trusted OIDC providers
# ToDo: use file?
flaat.set_trusted_OP_list([
'https://b2access.eudat.eu/oauth2/',
'https://b2access-integration.fz-juelich.de/oauth2',
'https://unity.helmholtz-data-federation.de/oauth2/',
'https://login.helmholtz-data-federation.de/oauth2/',
'https://login-dev.helmholtz.de/oauth2/',
'https://login.helmholtz.de/oauth2/',
'https://unity.eudat-aai.fz-juelich.de/oauth2/',
'https://services.humanbrainproject.eu/oidc/',
'https://accounts.google.com/',
'https://aai.egi.eu/oidc/',
'https://aai-dev.egi.eu/oidc/',
'https://login.elixir-czech.org/oidc/',
'https://iam-test.indigo-datacloud.eu/',
'https://iam.deep-hybrid-datacloud.eu/',
'https://iam.extreme-datacloud.eu/',
'https://oidc.scc.kit.edu/auth/realms/kit/',
'https://proxy.demo.eduteams.org'
])

# configuration for plotting
# ToDo: use file?
plot_conf = {
    'plot_t': 'type',
    'time_c': 'time',
    'tco': {
        'fig_size': [9, 6],
        'inputs' : ['begin_year', 'end_year', 'lat_min', 'lat_max']
        }
}


def _catch_error(f):
    """Decorate function to return an error, in case
    """
    # In general, API should return what is requested, i.e.
    # JSON -> JSON, PDF->PDF
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            e_message = []
            e_message.append({ 'status': 'Error',
                               'object': str(type(e)),
                               'message': '{}'.format(e)
                             })
            logger.debug(e_message)
            #raise BadRequest(e)

            if request.headers['Accept'] == "application/pdf":
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size = 14)
                for key, value in e_message[0].items():
                    pdf.cell(120, 20, txt = "{} : {}".format(key, value), 
                             ln = 1, align = 'L')
                
                pdf_byte_str = pdf.output(dest='S').encode('latin-1')
                buffer_resp = BytesIO(bytes(pdf_byte_str))
                buffer_resp.seek(0)

                response = make_response(send_file(buffer_resp,
                                         as_attachment=True,
                                         attachment_filename='Error.pdf',
                                         mimetype='application/pdf'), 500)
            else:
                response = make_response(jsonify(e_message), 500)
              
            logger.debug("Response: {}".format(dict(response.headers)))
            return response

    return wrap


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
                               concat_dim=plot_conf['time_c'],
                               data_vars='minimal', coords='minimal',
                               parallel=True)
    else:
        ds = xr.open_mfdataset(files,
                               concat_dim=plot_conf['time_c'],
                               data_vars='minimal', coords='minimal',
                               parallel=True)

    logger.info("Dataset is loaded from storage location: {}".format(ds))
    
    return ds


def get_date_range(ds):
    """Return range of dates in the provided data
    :param ds: xarray dataset to check
    :return: date_min, date_max
    """
    date_min = np.amin(ds.coords[plot_conf['time_c']].values)
    date_max = np.amax(ds.coords[plot_conf['time_c']].values)

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
    plot_type = kwargs[plot_conf['plot_t']]
    plot_title = plot_type + " (inputs: "
    for par in plot_conf[plot_type]['inputs']:
        plot_title += str(kwargs[par]) + ","

    plot_title = plot_title[:-1] + ")" # replace last "," with ")"
    
    return plot_title

    
def set_file_name(**kwargs):
    """Set file name
    :param kwargs: provided in the API call parameters
    :return: file_name with added input parameters (no extension given!)
    :rtype: string
    """
    plot_type = kwargs[plot_conf['plot_t']]
    file_name = plot_type
    for par in plot_conf[plot_type]['inputs']:
        file_name += "_" + str(kwargs[par])

    return file_name


def process(**kwargs):
    """Select data processing according to the plot type
    :param kwargs: provided in the API call parameters
    :return: processed data
    """

    plot_type = kwargs[plot_conf['plot_t']]
    data = kwargs['ds']
    
    if plot_type == 'tco':
        data_processed = o3plots.process_for_tco(**kwargs)
    elif plot_type == 'xx':
        data_processed = data
    else:
        # should return something
        data_processed = data

    return data_processed


@flaat.login_required() # Require only authorized people to call api method   
@_catch_error
def plot(**kwargs):
    """Main plotting routine
    :param kwargs: provided in the API call parameters
    :return: either PDF plot or JSON document
    """    
    time_start = time.time()

    json_output = []

    plot_type = kwargs[plot_conf['plot_t']]
    models = kwargs['models']

    logger.debug("headers: {}".format(dict(request.headers)))
    logger.debug("models: {}".format(models))
    
    if request.headers['Accept'] == "application/pdf":
        fig = plt.figure(num=None, figsize=(plot_conf[plot_type]['fig_size']), 
                         dpi=150, facecolor='w', 
                         edgecolor='k')
    else:
        fig_type = {"plot_type": plot_type}
        json_output.append(fig_type)
                         
    for model in models:
        time_model = time.time()
        # strip possible spaces in front and back
        model = model.lstrip().rstrip()
        logger.debug("model = {}".format(model))
        
        # get list of files for the model
        data_files = get_datafiles(model)
        
        # create dataset using xarray
        ds = get_dataset(data_files)
        kwargs['ds'] = ds
        
        # process data according to the plot type
        data_processed = process(**kwargs)
 
        time_described = time.time()
        logger.debug("[TIME] Processing described: {}".format(time_described - 
                                                              time_model))

        data_processed.load()
        time_loaded = time.time()
        logger.debug("[TIME] Processing finished: {}".format(time_loaded -
                                                             time_described))

        # convert to pandas series to keep date information
        time_axis = pd.DatetimeIndex(
                            data_processed.coords[plot_conf['time_c']].values)
        curve = pd.Series(data_processed[plot_type], 
                          index=time_axis,
                          name=model )

        # data visualisation, if pdf is asked for,
        # or add data points as json
        if request.headers['Accept'] == "application/pdf":
            curve.plot()
            periodicity = get_periodicity(time_axis)
            logger.info("Data periodicity: {} points/year".format(periodicity))
            decompose = seasonal_decompose(curve, period=periodicity)
            trend = pd.Series(decompose.trend, 
                              index=time_axis,
                              name=model+" (trend)" )
            trend.plot()

        else:
            observed = {"model": model,
                        "x": time_axis.tolist(),
                        "y": np.nan_to_num(data_processed[plot_type]).tolist(),
                   }
            json_output.append(observed)


    # finally return either PDF plot
    # or JSON document
    if request.headers['Accept'] == "application/pdf":
        figure_file = set_file_name(**kwargs) + ".pdf"
        plt.title(set_plot_title(**kwargs))
        plt.legend()
        buffer_plot = BytesIO()  # store in IO buffer, not a file
        plt.savefig(buffer_plot, format='pdf')
        plt.close(fig)
        buffer_plot.seek(0)

        response = send_file(buffer_plot,
                             as_attachment=True,
                             attachment_filename=figure_file,
                             mimetype='application/pdf')
    else: 
        #ToDo: should better differentiate json/pdf/anything
        #request.headers['Accept'] == "application/json":
        response = json_output


    logger.info(
       "[TIME] Total time from getting the request: {}".format(time.time() -
                                                               time_start))
    
    return response
