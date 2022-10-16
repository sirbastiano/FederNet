#LIBRERIE

import numpy as np 
import sympy as sym
from filterpy.kalman import KalmanFilter 
import scipy
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import matplotlib.pyplot as plt
from matplotlib import style
import pymap3d as pym
import pandas as pd
import astropy
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.twobody.propagation import cowell as cowell
from poliastro.core.perturbations import J3_perturbation, J2_perturbation
from poliastro.core.propagation import func_twobody
import glob
import math
from sklearn import linear_model, datasets
import matplotlib.style as style
import time
from tkinter import SW
from numpy.linalg import inv, det

from utilitys.utils import *
from utilitys.MieFunzionis import *

import cv2

###########################
#COSTANTI
dt = 10
mi = 4.9048695e3 # km^3/s^2 
S = 0.006 # m (focal length 600 mm per la wide angle camera)
FOV=61.4 #° WIDE ANGLE CAMERA


#############################
#DATI REALI
#Lat, Long, Alt
df = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\LLA.csv") 
real_Latitudes, real_Longitudes, real_Altitudes = df['Lat (deg)'], df['Lon (deg)'], df['Alt (km)']

#posizione
real_X, real_Y, real_Z = [], [], []
for i in range(len(df)):
    altitude = real_Altitudes[i]
    latitude = real_Latitudes[i]
    longitude = real_Longitudes[i]
    x, y, z = spherical2cartesian(altitude, latitude, longitude)
    real_X.append(x)
    real_Y.append(y-50)
    real_Z.append(z)
real_X, real_Y, real_Z = np.array(real_X),np.array(real_Y),np.array(real_Z)
#dpf = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\Position_Fixed.csv") 
#real_X1, real_Y1, real_Z1  = dpf['x (km)'], dpf['y (km)'],dpf['z (km)']


#database crateri
DB = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython1\orbite\lunar_crater_database_robbins_2018.csv") 


#"Scatto la foto"


H, latitude, longitude = real_Altitudes[0],real_Latitudes[0], real_Longitudes[0]
E, N, U = LCLF2ENU (real_X[0], real_Y[0], real_Z[0], latitude, longitude)
d = SW_nadir(H)
ES=E-0.5*d
ED=E+0.5*d
NS=N-0.5*d
NG=N+0.5*d
E_A = ES
N_A = NS
#Punto B
E_B = ED
N_B = NS
#Punto C
E_C = ES
N_C = NG
    #Punto D
E_D = ED
N_D = NG
                                         #(est,nord,up,lat,long,alt)
lat_a, long_a, alt_a=pym.enu2geodetic(E_A,N_A,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))

lat_b, long_b, alt_b=pym.enu2geodetic(E_B,N_B,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))

lat_c, long_c, alt_c=pym.enu2geodetic(E_C,N_C,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))

lat_d, long_d, alt_d=pym.enu2geodetic(E_D,N_D,U,latitude,longitude,-1787.4,ell=pym.utils.Ellipsoid('moon'))
    
phi_A=lat_a
phi_C=lat_c

lambd_A=long_a
lambd_B=long_b

if phi_A<phi_C:
    lat_inf1=phi_A
    lat_sup1=phi_C
else:         
    lat_inf1=phi_C
    lat_sup1=phi_A

if lambd_A<lambd_B:
    long_inf1=lambd_A
    long_sup1=lambd_B
else:         
    long_inf1=lambd_B
    long_sup1=lambd_A
    
lat_inf=lat_inf1
lat_sup=lat_sup1
long_sup=long_sup1
long_inf=long_inf1    
    
# Filtering DATABASE:
lat_bounds=[lat_inf, lat_sup]
lon_bounds=[long_inf,long_sup]
craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='ROBBINS')
crater_Latitudes, crater_Longitudes = craters_cat['Lat'], craters_cat['Lon']

indexNames = craters_cat[ (craters_cat['Diam'] <= 3) ].index
craters_cat.drop(indexNames , inplace=True)
indexNames = craters_cat[ (craters_cat['Diam'] >= 20) ].index
craters_cat.drop(indexNames , inplace=True)
craters_cat = craters_cat.reset_index(drop=True)
          
N_crat =len(craters_cat)


if N_crat>10:
    craters_cat=craters_cat.sample(n=10)
    craters_cat = craters_cat.reset_index(drop=True)
    N_crat=len(craters_cat)

crater_Latitudess, crater_Longitudess = craters_cat['Lat'], craters_cat['Lon']
lat_c, lon_c = [], []

for i in range(N_crat):
    altitude = 0
    latitudec = crater_Latitudess[i]
    longitudec = crater_Longitudess[i]

    x, y, z = spherical2cartesian(altitude, latitudec, longitudec)


    h_cr,lat_cr,long_cr=cartesian2spherical(x,y,z)

    lat_c.append(lat_cr)
    lon_c.append(long_cr)    
lat_c=np.array(lat_c)
lon_c=np.array(lon_c)
       
rectangleY = [lat_sup, lat_sup, lat_inf, lat_inf, lat_sup]
rectangleX = [long_inf, long_sup, long_sup, long_inf, long_inf]
plt.figure()  

plt.scatter(crater_Longitudes,crater_Latitudes,s=50,marker='x',color = 'r')
plt.plot(lon_c,lat_c,'o',color = 'g')
plt.plot(real_Longitudes[0],real_Latitudes[0],'o')
#plt.plot(long_,lat_,'o',color = 'b')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(("Real craters","Found craters","Central point"))
plt.plot(rectangleX, rectangleY, 'o')
plt.plot(rectangleX, rectangleY, '-')
plt.show()
