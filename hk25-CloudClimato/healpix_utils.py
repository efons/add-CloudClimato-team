import healpy as hp 
import numpy as np 
import xarray as xr


def get_nn_lon_lat_index(nside, lons, lats):
    # from https://easy.gems.dkrz.de/Processing/healpix/lonlat_remap.html 
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )