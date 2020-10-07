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
# tco3_zm: total column ozone, zonal mean

# ToDo: improve Error handling, that Errors are correctly returned by API
#       e.g. raise OSError("no files to open")

import o3as.config as cfg
import o3as.plothelpers as phlp
import o3as.plots as o3plots
import json
import logging
import matplotlib.pyplot as plt
import numpy as np

import os
import pkg_resources
import pandas as pd
import time

from flask import send_file
from flask import jsonify, make_response, request
from fpdf import FPDF
from functools import wraps
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose # accurate enough

# conigure python logger
logger = logging.getLogger('__name__') #o3asplot
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s')
logger.setLevel(cfg.log_level)

## Authorization
from flaat import Flaat
flaat = Flaat()

# list of trusted OIDC providers
flaat.set_trusted_OP_list(cfg.trusted_OP_list)

# configuration for plotting
pconf = cfg.plot_conf


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
                    pdf.write(18, txt = "{} : {}".format(key, value))
                    pdf.ln()
                
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

@_catch_error
def get_metadata(*args, **kwargs):
    """Return information about API
    :return: json with info
    """
    module = __name__.split('.', 1)
    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'name' : None,
        'version' : None,
        'summary' : None,
        'home-page' : None,
        'author' : None,
        'author-email' : None,
        'license' : None
    }
    iline = 0
    top_lines = 10 # take only top 10 lines (otherwise may pick from content)
    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
        if iline < top_lines:
            for par in meta:
                if line_low.startswith(par.lower() + ":", 0):
                    _, value = line.split(": ", 1)
                    meta[par] = value
        iline += 1
    
    logger.debug(F"Found metadata: {meta}")    
    return meta

@_catch_error
def list_models(*args, **kwargs):
    """Return list of available models
    :return: list of models
    """
    models = []
    for mdir in os.listdir(cfg.O3AS_DATA_BASEPATH):
        m_path = os.path.join(cfg.O3AS_DATA_BASEPATH, mdir) 
        if (os.path.isdir(m_path)):
            netcdf_ok = False
            for f in os.listdir(m_path):
                if ".nc" in f:
                    netcdf_ok = True
            models.append(mdir) if netcdf_ok else ''
    
    models.sort()
    models_dict = { "models": models }
    logger.debug(F"Model list: {models_dict}")

    return models_dict

@_catch_error
def get_model_info(*args, **kwargs):
    """Return information about a model
    :return: info about a model
    """
    model = kwargs['model'].lstrip().rstrip()

    # get list of files for the model
    data_files = phlp.get_datafiles(model)    

    # create dataset using xarray
    ds = phlp.get_dataset(data_files)
      
    info_dict = ds.to_dict(data=False)
    info_dict['model'] = model

    logger.debug(F"{model} model info: {info_dict}")
    return info_dict


@flaat.login_required() # Require only authorized people to call api method   
@_catch_error
def plot(*args, **kwargs):
    """Main plotting routine
    :param kwargs: provided in the API call parameters
    :return: either PDF plot or JSON document
    """    
    time_start = time.time()

    json_output = []

    plot_type = kwargs[pconf['plot_t']]
    models = kwargs['models']

    logger.debug(F"headers: {dict(request.headers)}")
    logger.debug(F"models: {models}")
    
    if request.headers['Accept'] == "application/pdf":
        fig = plt.figure(num=None, figsize=(pconf[plot_type]['fig_size']), 
                         dpi=150, facecolor='w', 
                         edgecolor='k')
    else:
        fig_type = {"plot_type": plot_type}
        json_output.append(fig_type)

    if plot_type == 'tco3_zm':
        data = o3plots.ProcessForTCO3(**kwargs)
    elif plot_type == 'vmro3_zm':
        data = o3plots.ProcessForVMRO3(**kwargs)
    elif plot_type == 'tco3_return':
        data = o3plots.ProcessForTCO3Return(**kwargs)
                         
    for model in models:
        time_model = time.time()
        # strip possible spaces in front and back
        model = model.lstrip().rstrip()
        logger.debug(F"model = {model}")

        # get data for the plot
        data_processed = data.get_plot_data(model)
 
        time_described = time.time()
        logger.debug("[TIME] Processing described: {}".format(time_described - 
                                                              time_model))

        data_processed.load()
        time_loaded = time.time()
        logger.debug("[TIME] Processing finished: {}".format(time_loaded -
                                                             time_described))

        # convert to pandas series to keep date information
        if (type(data_processed.indexes[pconf['time_c']]) is 
            pd.core.indexes.datetimes.DatetimeIndex) :
            time_axis = data_processed.indexes[pconf['time_c']].values
        else:
            time_axis = data_processed.indexes[pconf['time_c']].to_datetimeindex()

        curve = pd.Series(np.nan_to_num(data_processed[plot_type]), 
                          index=pd.DatetimeIndex(time_axis),
                          name=model )

        # data visualisation, if pdf is asked for,
        # or add data points as json
        if request.headers['Accept'] == "application/pdf":
            curve.plot()
            periodicity = phlp.get_periodicity(time_axis)
            logger.info("Data periodicity: {} points/year".format(periodicity))
            decompose = seasonal_decompose(curve, period=periodicity)
            trend = pd.Series(decompose.trend, 
                              index=time_axis,
                              name=model+" (trend)" )
            trend.plot()

        else:
            observed = {"model": model,
                        "x": curve.index.tolist(),
                        "y": curve.values.tolist(),
                   }
            json_output.append(observed)


    # finally return either PDF plot
    # or JSON document
    if request.headers['Accept'] == "application/pdf":
        figure_file = phlp.set_file_name(**kwargs) + ".pdf"
        plt.title(phlp.set_plot_title(**kwargs))
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
