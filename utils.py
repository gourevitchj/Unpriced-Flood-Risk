### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: utils.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os
import scipy
import pickle
import binsreg
import numpy as np
import pandas as pd

from pandas.api import types
from twilio.rest import Client

import paths

def send_textmessage(txt_msg):

    """Send text message to phone.

    Arguments:
        txt_msg (string): Message to send
                
    Returns:
        None
    """

    ### Activate client using account SID and authentication token
    account_sid = 'REDACTED' 
    auth_token = 'REDACTED' 
    client = Client(account_sid, auth_token) 

    ### Send text message to phone
    client.messages.create(
        to='REDACTED', 
        from_='REDACTED', 
        body=txt_msg)

    return None


class CustomUnpickler(pickle.Unpickler):

    ### Get data from pickle file
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


def read_pickle(pickle_uri):

    """Read pickle file.

    Arguments:
        pickle_uri (string): Path to pickle file
                
    Returns:
        data (unknown): Whatever is contained within the pickle
    """

    ### Get data from pickle file
    data = CustomUnpickler(open(pickle_uri, 'rb')).load()
    
    return data


def npv(df, eal_col_2020, eal_col_2050, discount_rate, time_horizon=30):

    """Calculate net present value.

    Arguments:
        df (Pandas Dataframe): Dataframe containing EAL columns
        eal_col_2020 (string): Name of column containing 2020 EAL data
        eal_col_2050 (string): Name of column containing 2050 EAL data
        discount_rate (float): Applied discount rate

    Optional arguments: 
        time_horizon (int): Number of years over which to calculate net present value
                
    Returns:
        npv (Pandas Series): Series of net present values
    """
    
    ### Create column to store NPV outputs by copying EAL 2020 column
    npv = df[eal_col_2020].copy()

    ### If discount rate is greater than one, convert it to decimal
    if discount_rate >= 1:
        discount_rate /= 100.0

    ### Calculate annual change in EAL between 2020 and 2050
    annual_eal_change = (df[eal_col_2050] - df[eal_col_2020]) / 30.0

    ### Iterate through years in time horizon
    for t in range(1, time_horizon+1):
      
        ### Calculate expected annual damages for time t
        ead = df[eal_col_2020] + (annual_eal_change * t)
                    
        ### Discount expected annual damages in time t
        ead_discounted = ead * ((1+discount_rate) ** (-1.0*t))
        
        ### Add discounted annual damages to net present value
        npv += ead_discounted

    return npv


def get_modeloutputs():

   """Get hedonic panel model outputs.

    Arguments:
        None

    Returns:
        model_a (Pandas Dataframe): Model A outputs
        model_b (Pandas Dataframe): Model B outputs
        model_c (Pandas Dataframe): Model C outputs
        model_d (Pandas Dataframe): Model D outputs
    """

    ### Read statistics outputs CSV to Pandas dataframe
    stats_csv_fn = 'HedonicOutputs_Summary.csv'
    stats_csv_uri = os.path.join(paths.outputs_dir, stats_csv_fn)
    df_stats = pd.read_csv(stats_csv_uri)

    ### Get model labels
    model_label_a = 'attitude-%s_coastal-x_disclosure-%s' %('0', '0')
    model_label_b = 'attitude-%s_coastal-x_disclosure-%s' %('0', '1')
    model_label_c = 'attitude-%s_coastal-x_disclosure-%s' %('1', '0')
    model_label_d = 'attitude-%s_coastal-x_disclosure-%s' %('1', '1')

    ### Get model data
    model_a = df_stats[df_stats['model_label']==model_label_a]
    model_b = df_stats[df_stats['model_label']==model_label_b]
    model_c = df_stats[df_stats['model_label']==model_label_c]
    model_d = df_stats[df_stats['model_label']==model_label_d]

    ### Return model data
    return model_a, model_b, model_c, model_d


def get_xs_coeff(flood_zone, a, d):

   """Get cross-sectional model coefficients.

    Arguments:
        flood_zone (string): Flood zone (either '100' ,'500', or 'outside') 
        a (int): Climate risk attitude (either 0 or 1)
        d (int): Disclsoure laws (either 0 or 1)

    Returns:
        coeff (int): Cross-sectional model coefficient
    """

    ### Read CSV file with estimated discounts from XS model
    df = pd.read_csv(paths.xs_coeffs_csv_uri)

    ### Subset dataframe to only include coefficients for at-risk properties
    df = df[df['risk']==1]

    ### Get group code
    if a == 0 and d == 0:
        group = 'ncnd'

    if a == 1 and d == 0:
        group = 'cnd'

    if a == 0 and d == 1:
        group = 'ncd'

    if a == 1 and d == 1:
        group = 'cd'

    ### Subset dataframe to group
    df = df[df['group']==group]

    ### Subset dataframe to floodzone
    df = df[df['fz_risk']==flood_zone]

    ### Get coeff
    coeff = float(df['discount'])

    return coeff


def binscatter(**kwargs):
        
   """Get cross-sectional model coefficients.

    Arguments:
        kwargs:
            x (string): Name of x variable 
            y (string): Name of y variable 
            data (Pandas Dataframe): Dataframe containing x and y variables 
            ci (tuple): confidence interval

    Returns:
        df_est (Pandas Dataframe): 
    """

    ### Estimate binsreg
    est = binsreg.binsreg(**kwargs)
    
    ### Retrieve estimates
    df_est = pd.concat([d.dots for d in est.data_plot])
    df_est = df_est.rename(columns={'x': kwargs.get("x"), 'fit': kwargs.get("y")})
    
    ### Add confidence intervals
    if "ci" in kwargs:
        df_est = pd.merge(df_est, pd.concat([d.ci for d in est.data_plot]))
        df_est = df_est.drop(columns=['x'])
        df_est['ci'] = df_est['ci_r'] - df_est['ci_l']
    
    ### Rename groups
    if "by" in kwargs:
        df_est['group'] = df_est['group'].astype(df[kwargs.get("by")].dtype)
        df_est = df_est.rename(columns={'group': kwargs.get("by")})

    return df_est


def bivariate_color(sx, sy, cmap, xbins, ybins, xlims=None, ylims=None):

    """Creates a color series for a combination of two series.
    
    Arguments:
        sx (Pandas Series): x-axis data
        sy (Pandas Series): y-axis data
        cmap (Numpy Array): Two-dimensional colormap
        xbins (list): Bins for the x-axis
        ybins (list): Bins for the y-axis

    Optional arguments:
        xlims (tuple): Optional tuple specifying limits to the x-axis
        ylims (tuple): Optional tuple specifying limits to the y-axis
    
    Returns:
        colors (Pandas Series): Series of assigned colors per cmap provided
    """

    x_numeric = types.is_numeric_dtype(sx)
    y_numeric = types.is_numeric_dtype(sy)
    x_categorical = types.is_categorical_dtype(sx)
    y_categorical = types.is_categorical_dtype(sy)

    msg = (
        "The provided {s} is not numeric or categorical. If {s} contains "
        "categories, transform the series to (ordered) pd.Categorical first."
    )
    if not x_numeric and not x_categorical:
        raise TypeError(msg.format(s="sx"))
    if not y_numeric and not y_categorical:
        raise TypeError(msg.format(s="sy"))

    ### If categorical, the number of categories have to equal the cmap shape.
    if x_categorical:
        if len(sx.categories) != cmap.shape[1]:
            raise ValueError(
                f"Length of x-axis colormap ({cmap.shape[1]}) does not match "
                f"the length of categories in sx ({len(sx.categories)}). "
                "Adjust the n of your cmap."
            )
    if y_categorical:
        if len(sy.categories) != cmap.shape[0]:
            raise ValueError(
                f"Length of x-axis colormap ({cmap.shape[0]}) does not match "
                f"the length of categories in sy ({len(sy.categories)}). "
                "Adjust the n of your cmap."
            )

    ### If numeric, use min/max to mock a series for the bins.
    if x_numeric:
        xmin, xmax = (sx.min(), sx.max()) if xlims is None else xlims
        if xbins is None:
            _, xbins = pd.cut(
                pd.Series([xmin, xmax]), cmap.shape[1], retbins=True
            )
    else:
        if xlims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sx: the xticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if xbins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sx: the xticks of the "
                "cmap are indivisible."
            )

    if y_numeric:
        ymin, ymax = (sy.min(), sy.max()) if ylims is None else ylims
        if ybins is None:
            _, ybins = pd.cut(
                pd.Series([ymin, ymax]), cmap.shape[0], retbins=True
            )
    else:
        if ylims is not None:
            raise RuntimeError(
                "Cannot apply limits to a categorical sy: the yticks of the "
                "cmap are indivisible. Instead, limit your data to the "
                "categories and adjust the n of cmap accordingly."
            )
        if ybins is not None:
            raise RuntimeError(
                "Cannot apply bins to a categorical sy: the yticks of the "
                "cmap are indivisible."
            )

    def _bin_value(x, bins):
        if bins.min() >= x:
            return 0  # First index.
        if bins.max() <= x:
            return len(bins[:-1]) - 1  # Last index.
        for i, v in enumerate(bins[:-1]):
            rangetest = v < x <= bins[i + 1]  # pd.cut right=True by default.
            if rangetest:
                return i
        return np.nan

    def _return_color(x, y, cmap):
        if np.isnan(x) or np.isnan(y):
            return (0.0, 0.0, 0.0, 0.0)  # Transparent white if one is np.nan.
        xidx = _bin_value(x, xbins) if x_numeric else x
        yidx = _bin_value(y, ybins) if y_numeric else y
        
        return tuple(cmap[yidx, xidx])

    sx = pd.Series(sx.codes) if x_categorical else sx
    sy = pd.Series(sy.codes) if y_categorical else sy

    df = pd.DataFrame([sx, sy]).T
    colors = df.apply(
        lambda g: _return_color(g[df.columns[0]], g[df.columns[1]], cmap),
        axis=1,
    )

    return colors

