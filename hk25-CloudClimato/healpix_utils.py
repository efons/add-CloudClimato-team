import healpy as hp 
import numpy as np 
import xarray as xr
import intake
import cartopy.crs as ccrs
import cartopy.feature as cf
import healpy as hp

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

def get_nn_lon_lat_index(nside, lons, lats):
    # from https://easy.gems.dkrz.de/Processing/healpix/lonlat_remap.html 
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )

def worldmap(var, colorbar_label, fig = None, ax = None, **kwargs):

    if ax == None:
        projection = ccrs.Robinson(central_longitude=-135.5808361)
        fig, ax = plt.subplots(
            figsize=(8, 6), subplot_kw={"projection": projection}, constrained_layout=True
        )
        ax.set_global()

    imsh = egh.healpix_show(var, ax=ax, **kwargs)
    
    cb = fig.colorbar(imsh, ax = ax, label=colorbar_label, orientation="horizontal", fraction = 1, pad = 0.01)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.add_feature(cf.COASTLINE, linewidth=1, color = 'w')
    # ax.add_feature(cf.BORDERS, linewidth=0.4)

def calc_albedo(ds_in, ds_out = None, in_sw_var = 'rsdt', out_sw_var = 'rsut', tag = ''):
    """
    albedo = incoming/outgoing
    """

    if ds_out == None:
        ds_out =  ds_in
        
    ds_out['albedo' + tag] = ds_in[out_sw_var] / ds_in[in_sw_var].where(ds_in[in_sw_var] != 0)
    return ds_out
    

def calc_lat_profile(ds_in, ds_out = None, var = None):
    
    if ds_out == None:
        ds_out =  ds_in

    
    if isinstance(var, str):
        ds_out[var + '_lat_prof'] = ds_in[var].mean('lon')
    elif var is None:
        for var in ds.data_vars:
            ds_out[var + '_lat_prof'] = ds_in[var].mean('lon')
    else:
        for vari in var:
            ds_out[vari + '_lat_prof'] = ds_in[vari].mean('lon')

    return ds_out


from scipy.constants import R as R_universal

def calc_lts(ds, temp_var='ta', p_lev_dim='level', p_unit='Pa', out_var='lts'):
    """
    Calculate Lower Tropospheric Stability (LTS = θ_700 - θ_1000) and add it to the dataset.

    Parameters:
        ds (xr.Dataset): Input dataset with temperature and pressure level dimension.
        temp_var (str): Name of the temperature variable (Kelvin).
        p_lev_dim (str): Name of pressure level coordinate (e.g., 'level', 'plev').
        p_unit (str): Unit of pressure levels: 'Pa' or 'hPa'.
        out_var (str): Name for the output LTS variable in the dataset.

    Returns:
        xr.Dataset: Dataset with added 'lts' variable.
    """
    # Constants
    M_dry = 0.0289652  # kg/mol
    R = R_universal / M_dry  # J/(kg·K) ≈ 287.05
    c_p = 1004.0  # J/(kg·K)
    kappa = R / c_p  # ≈ 0.286

    # Define target pressures
    if p_unit == 'Pa':
        p700, p1000 = 70000, 100000
    elif p_unit == 'hPa':
        p700, p1000 = 700, 1000
    else:
        raise ValueError("p_unit must be 'Pa' or 'hPa'")

    # Select temperature at 700 and 1000 hPa (nearest match)
    T_700 = ds[temp_var].sel({p_lev_dim: p700}, method='nearest')
    T_1000 = ds[temp_var].sel({p_lev_dim: p1000}, method='nearest')

    p_ref = p1000
    # Compute potential temperatures
    theta_700 = T_700 * (p_ref / p700) ** kappa
    theta_1000 = T_1000 * (p_ref / p1000) ** kappa

    # Calculate LTS and add to dataset
    lts = theta_700 - theta_1000
    lts.attrs['units'] = 'K'
    lts.attrs['long_name'] = 'Lower Tropospheric Stability (θ_700 - θ_1000)'

    ds[out_var] = lts
    return ds




def get_nn_data(var, nx=1000, ny=1000, ax=None):
    """
    var: variable (array-like)
    nx: image resolution in x-direction
    ny: image resolution in y-direction
    ax: axis to plot on
    returns: values on the points in the plot grid.
    """
    lonlat = get_lonlat_for_plot_grid(nx, ny, ax)
    try:
        return get_healpix_nn_data(var, lonlat)
    except ValueError:
        pass
    if set(var.dims) == {"lat", "lon"}:
        return get_lonlat_meshgrid_nn_data(var, lonlat)
    else:
        return get_lonlat_nn_data(var, lonlat)


def get_healpix_nn_data(var, lonlat):
    """
    var: variable on healpix coordinates (array-like)
    lonlat: coordinates at which to get the data
    returns: values on the points in the plot grid.
    """
    valid = np.all(np.isfinite(lonlat), axis=-1)
    points = lonlat[valid].T  # .T reverts index order
    pix = hp.ang2pix(
        hp.npix2nside(len(var)), theta=points[0], phi=points[1], nest=True, lonlat=True
    )
    res = np.full(lonlat.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]
    return res


def get_lonlat_nn_data(var, lonlat):
    """
    var: variable with lon and lat attributes (2d slice)
    lonlat: coordinates at which to get the data
    returns: values on the points in the plot grid.
    """
    var_xyz = lonlat_to_xyz(lon=var.lon.values.flatten(), lat=var.lat.values.flatten())
    tree = KDTree(var_xyz)

    valid = np.all(np.isfinite(lonlat), axis=-1)
    ll_valid = lonlat[valid].T
    plot_xyz = lonlat_to_xyz(lon=ll_valid[0], lat=ll_valid[1])

    distances, inds = tree.query(plot_xyz)
    res = np.full(lonlat.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var.values.flatten()[inds]
    return res


def get_lonlat_meshgrid_nn_data(var, lonlat):
    """
    var: variable with lon and lat attributes (2d slice)
    lonlat: coordinates at which to get the data
    returns: values on the points in the plot grid.
    """
    return get_lonlat_nn_data(var.stack(cell=("lon", "lat")), lonlat)


def get_lonlat_for_plot_grid(nx, ny, ax=None):
    """
    nx: image resolution in x-direction
    ny: image resolution in y-direction
    ax: axis to plot on
    returns: coordinates of the points in the plot grid.
    """

    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    xvals2, yvals2 = np.meshgrid(xvals, yvals)
    lonlat = ccrs.PlateCarree().transform_points(
        ax.projection, xvals2, yvals2, np.zeros_like(xvals2)
    )
    return lonlat


def lonlat_to_xyz(lon, lat):
    """
    lon: longitude in degree E
    lat: latitude in degree N
    returns numpy array (3, len (lon)) with coordinates on unit sphere.
    """

    return np.array(
        (
            np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lat)),
        )
    ).T

def plot_map_diff(var, ref, colorbar_label="", title="", extent=None, **kwargs):
    """
    var: data set
    ref: reference data
    colorbar_label: label for the colorbar
    title: title string
    **kwargs: get passed to imshow
    returns figure, axis objects
    """
    projection = ccrs.Robinson(central_longitude=-135.5808361)
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    varmap = get_nn_data(var, ax=ax)
    refmap = get_nn_data(ref, ax=ax)
    imsh = ax.imshow(
        varmap - refmap, extent=ax.get_xlim() + ax.get_ylim(), origin="lower", **kwargs
    )
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    fig.colorbar(imsh, label=colorbar_label)
    plt.title(title)
    return (fig, ax)