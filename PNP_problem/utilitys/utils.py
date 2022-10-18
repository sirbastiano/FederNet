from numba import njit
from astropy.coordinates.funcs import spherical_to_cartesian, cartesian_to_spherical
import numpy as np
import pandas as pd
#import cv2
from copy import deepcopy
import glob
import math




def cartesian2spherical(x, y, z):
    ###########################################################################
    # Inputs: x, y, z
    ###########################################################################
    # Outputs:
    ###########################################################################
    # h: Altitude (km)
    # Lat: Latitude (deg)
    # Lon: Longitude (deg)
    ###########################################################################
    h, Lat, Lon = cartesian_to_spherical(x, y, z)
    R_moon = 1737.4
    h = h - R_moon
    Lon = np.where(Lon.deg < 180, Lon.deg, Lon.deg - 360)
    Lat = np.where(Lat.deg < 90, Lat.deg, Lat.deg - 360)
    return np.array(h), np.array(Lat), np.array(Lon)


def spherical2cartesian(h, Lat, Lon):
    ###########################################################################
    # Inputs:
    ###########################################################################
    # h: Altitude (km)
    # Lat: Latitude (deg)
    # Lon: Longitude (deg)
    ###########################################################################
    # Outputs: x, y, z
    ###########################################################################
    R_moon = 1737.4
    x, y, z = spherical_to_cartesian(
        h + R_moon, np.deg2rad(Lat), np.deg2rad(Lon))
    return -np.array(x), -np.array(y), -np.array(z)


def CatalogSearch(H, lat_bounds: np.array, lon_bounds: np.array, CAT_NAME):
    # -180 to 180 // formulation 1
    #   0  to 360 // formulation 2
    # Example: 190 lon //formulation 2 --> -170 lon // formulation 1
    # -10 lon == 350 lon
    # We want to pass from f1 --> f2
    if CAT_NAME == "LROC":
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diameter (km)"])
        LONs = np.array(H["Long"])

    elif CAT_NAME == "HEAD":
        LONs = np.array(H["Lon"])
        LATs = np.array(H["Lat"])
        DIAMs = np.array(H["Diam_km"])
    elif CAT_NAME == "ROBBINS":
        LONs = np.array(H["LON_CIRC_IMG"])
        LATs = np.array(H["LAT_CIRC_IMG"])
        DIAMs = np.array(H["DIAM_CIRC_IMG"])

    elif CAT_NAME == "COMBINED":
        LONs = np.array(H["lon"])
        LATs = np.array(H["lat"])
        DIAMs = np.array(H["diam"])

    LONs_f1 = np.where(LONs > 180, LONs - 360, LONs)

    cond1 = LONs_f1 < lon_bounds[1]
    cond2 = LONs_f1 > lon_bounds[0]
    cond3 = LATs > lat_bounds[0]
    cond4 = LATs < lat_bounds[1]

    filt = cond1 & cond2 & cond3 & cond4

    LATs = LATs[filt]
    LONs_f1 = LONs_f1[filt]
    DIAMs = DIAMs[filt]
    if LONs_f1 != []:
        craters = np.hstack(
            [np.vstack(LONs_f1), np.vstack(LATs), np.vstack(DIAMs)])
        df = pd.DataFrame(data=craters, columns=["Lon", "Lat", "Diam"])
        return df
    else:
        pass


