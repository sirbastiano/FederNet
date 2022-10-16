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

def rotation_matrix_to_attitude_angles(R):
    import math
    import numpy as np 
    cos_beta = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
    validity = cos_beta < 1e-6
    if not validity:
        alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = math.atan2(R[2,1], R[2,2])    # roll  [x]
    else:
        
        alpha = 0   # yaw   [z]
        beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
        gamma = 0                             # roll  [x]  
    return np.array([alpha, beta, gamma]) 

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
    real_Y.append(y)
    real_Z.append(z)
real_X, real_Y, real_Z = np.array(real_X),np.array(real_Y),np.array(real_Z)

dpf = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\Position_Fixed.csv") 
#real_X, real_Y, real_Z  = dpf['x (km)']*(-1), dpf['y (km)']*(-1),dpf['z (km)']*(-1)

#angoli di Eulero
dq = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\Attitude_Quaternions_Fixed.csv")  
real_t1 = dq['yaw (deg)']
real_t2 = dq['pitch (deg)']  
real_t3 = dq['roll (deg)']
#velocità angolari
real_om1, real_om2, real_om3 = [],[],[]
for i in range(len(dq)-2):
    om1 = (real_t1[i+1]-real_t1[i])/10
    om2 = (real_t2[i+1]-real_t2[i])/10
    om3 = (real_t3[i+1]-real_t3[i])/10
    
    real_om1.append(om1)
    real_om2.append(om2)
    real_om3.append(om3)
real_om1, real_om2, real_om3 = np.array(real_om1),np.array(real_om2),np.array(real_om3)

#velocita' 
dvf = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython2\orbite\Orbit1_nadir\Orbit1_nadir\Velocity_Fixed.csv") 
real_Vxs,real_Vys,real_Vzs = dvf['vx (km/sec)']*(-1), dvf['vy (km/sec)']*(-1),dvf['vz (km/sec)']*(-1)

#Condizioni iniziali
init_x, init_y, init_z = real_X[0], real_Y[0], real_Z[0]
init_vx, init_vy, init_vz = real_Vxs[0], real_Vys[0], real_Vzs[0]
init_teta1, init_teta2, init_teta3 = real_t1[0], real_t2[0], real_t3[0]
init_om1, init_om2, init_om3 = real_om1[0], real_om2[0], real_om3[0]

#database crateri
DB = pd.read_csv(r"C:\Users\formi\OneDrive\Desktop\KalmanPython1\orbite\lunar_crater_database_robbins_2018.csv") 

X=np.array((init_x, init_y, init_z)).T
Tet=np.array([[init_teta1, init_teta2, init_teta3]]).T

    
#############################

Ne=[]
trans = []
rot = []
#for i in range(len(df)):    
for i in range(10):
    #print("Iterazione ", i)
    #"Scatto la foto"
    step=1    
    H, latitude, longitude = real_Altitudes[i],real_Latitudes[i], real_Longitudes[i]
    #print("Ground truth (altitude, latitude, longitude): ",H, latitude, longitude  )
    E, N, U = LCLF2ENU(real_X[i], real_Y[i], real_Z[i], latitude, longitude)
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


    if craters_cat is None:  
        Ne.append(0)

    else: 

        indexNames = craters_cat[ (craters_cat['Diam'] <= 0) ].index
        craters_cat.drop(indexNames , inplace=True)
        indexNames = craters_cat[ (craters_cat['Diam'] >= 20) ].index
        craters_cat.drop(indexNames , inplace=True)
        craters_cat = craters_cat.reset_index(drop=True)
          
        N_crat =len(craters_cat)
            
        if N_crat>40:
            craters_cat=craters_cat.sample(n=40,random_state=1)
            craters_cat = craters_cat.reset_index(drop=True)
            N_crat=len(craters_cat)

        else: 
            pass

        print(" ") 

        if N_crat==0:
            Ne.append(0)
        else:     
            Ne.append(N_crat)   
            crater_Latitudes, crater_Longitudes = craters_cat['Lat'], craters_cat['Lon']

            x_c, y_c, z_c = [], [], []
            coo2=np.empty([0,2])
            coo3=np.empty([0,3])
            for i in range(N_crat):
                altitude = H
                latitudec = crater_Latitudes[i]
                longitudec = crater_Longitudes[i]

                x, y, z = spherical2cartesian(altitude, latitudec, longitudec)
                x_c.append(x*(0.001))
                y_c.append(y*(0.001))
                z_c.append(z*(0.001))
                xyz_i=np.array((x,y,z))
                coo3 = np.vstack((coo3, xyz_i))

                u=(longitudec*1024)/long_sup
                v=((90-latitudec)*1024)/(90-lat_inf)
                

                coo_i=np.array((u,v))
                coo2 = np.vstack((coo2, coo_i))
                
            coo_2D=coo2 #coordinate 2D
            coo_3D=coo3 #coordinate 3D
            #print("Coordinate 2D crateri: ", coo_2D)
            x_c, y_c, z_c = np.array(x_c),np.array(y_c),np.array(z_c)

            cx=(longitude*1024)/long_sup #coordinate pixel principal point
            cy=((90-latitude)*1024)/(90-lat_inf)
            #print("coordinate 2D ground truth: ", cx, " ", cy)
            
            K_matrix=np.array([(S, 0, cx),(0,S,cy),(0,0,1)])  #camera matrix
            dist_coeffs = np.zeros((4,1))

            #DIVERSI PNP SOLVER
            #success, rotation_vector, translation_vector = cv2.solvePnP(coo_3D, coo_2D, K_matrix, dist_coeffs, flags=0)
            #success, rotation_vector, translation_vector = cv2.solvePnP(coo_3D, coo_2D, K_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE) #require at least 4 points
            #success, rotation_vector, translation_vector = cv2.solvePnP(coo_3D, coo_2D, K_matrix, dist_coeffs, flags=cv2.SOLVEPNP_AP3P) #require only 4 points
            success, rotation_vector, translation_vector = cv2.solvePnP(coo_3D, coo_2D, K_matrix, dist_coeffs, useExtrinsicGuess = True,flags=cv2.SOLVEPNP_EPNP) #no n limits
            #success, rotation_vector, translation_vector = cv2.solvePnP(coo_3D, coo_2D, K_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP) #require at least 3 points

            rmat = cv2.Rodrigues(rotation_vector)[0]
            camera_position = -np.matrix(rmat).T * np.matrix(translation_vector)
            position=camera_position.T
            trans.append(position)
            att=rotation_matrix_to_attitude_angles(rmat)
            te1=math.degrees(-att[0]-0.785398)
            if te1<0:
                te1=360+te1
            te1=te1
            te2=math.degrees(-att[1])
            te3=math.degrees(-att[2])
            attitude=np.array((te1,te2,te3))
            rot.append(attitude)
trans=np.array(trans)
rot=np.array(rot)

plt.figure()
plt.title('Numero di crateri individuati')
plt.plot(Ne, marker="o", linestyle="")
plt.show(block=False)

x_pred = []
y_pred = []
z_pred = []


for t in trans:
    x = t[0,0]
    x_pred.append(x)

    y = t[0,1]
    y_pred.append(y)
       
    z =t[0,2]
    z_pred.append(z)

x_true = real_X[:len(x_pred)]
y_true = real_Y[:len(y_pred)] 
z_true = real_Z[:len(z_pred)]

lw=1

plt.figure(dpi=150, tight_layout=True)
#plt.figure()
plt.subplot(3,1,1) 
plt.plot(x_pred, '-k', linewidth=lw)
plt.plot(x_true, 'r', linewidth=lw)
plt.legend(("PnP","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('X [Km]')

plt.subplot(3,1,2) 
plt.plot(y_pred, '-k', linewidth=lw)
plt.plot(y_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Y [Km]')

plt.subplot(3,1,3) 
plt.plot(z_pred, '-k', linewidth=lw)
plt.plot(z_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Z [Km]')
plt.show(block=False)

plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
x_pred = np.array(x_pred)
x_true = np.array(x_true)
diff_x = []
for x,y in zip(x_pred,x_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along X ')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(312)
y_pred = np.array(y_pred)
y_true = np.array(y_true)
diff_y = []
for x,y in zip(y_pred,y_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along Y ')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(313)
z_pred = np.array(z_pred)
z_true = np.array(z_true)
diff_z = []
for x,y in zip(z_pred,z_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along Z ')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km')
plt.show(block=False)

t1_pred = []
t2_pred = []
t3_pred = []


for r in rot:
    t1 = r[0]
    t1_pred.append(t1)

    t2 = r[1]
    t2_pred.append(t2)
       
    t3 =r[2]
    t3_pred.append(t3)
     

t1_true = real_t1[:len(t1_pred)]
t2_true = real_t2[:len(t2_pred)] 
t3_true = real_t3[:len(t3_pred)]

lw=1

plt.figure(dpi=150, tight_layout=True)
#plt.figure()
plt.subplot(3,1,1) 
plt.plot(t1_pred, '-k', linewidth=lw)
plt.plot(t1_true, 'r', linewidth=lw)
plt.legend(("PnP","Reale"))
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta 1')

plt.subplot(3,1,2) 
plt.plot(t2_pred, '-k', linewidth=lw)
plt.plot(t2_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta 2')

plt.subplot(3,1,3) 
plt.plot(t3_pred, '-k', linewidth=lw)
plt.plot(t3_true, 'r', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Teta 3')
plt.show(block=False)

plt.figure(dpi=180, tight_layout=True)
plt.subplot(311)
t1_pred = np.array(t1_pred)
t1_true = np.array(t1_true)
diff_x = []
for x,y in zip(t1_pred,t1_true):
    d = (x - y)
    diff_x.append(d)
plt.title('Error along teta1 ')
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('deg')

plt.subplot(312)
t2_pred = np.array(t2_pred)
t2_true = np.array(t2_true)
diff_y = []
for x,y in zip(t2_pred,t2_true):
    d = (x - y)
    diff_y.append(d)
plt.title('Error along teta2 ')
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('deg')

plt.subplot(313)
t3_pred = np.array(t3_pred)
t3_true = np.array(t3_true)
diff_z = []
for x,y in zip(t3_pred,t3_true):
    d = (x - y)
    diff_z.append(d)
plt.title('Error along teta3 ')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('deg')
plt.show()

