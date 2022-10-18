#LIBRERIE
import pickle
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
import seaborn as sns
sns.set()

from utilitys.utils import *
from utilitys.MieFunzionis import *

import cv2

###########################
#COSTANTI
dt = 10
mi = 4.9048695e3 # km^3/s^2 
S = 0.006 # m (focal length 600 mm per la wide angle camera)
FOV=61.4 #° WIDE ANGLE CAMERA

SFT = 0

def rotation_matrix_to_attitude_angles(R):
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

def find_diff(a,b): 
    return np.array([x-y if x is not None else None for x,y in zip(a,b)])

deg2km=(2*math.pi*1737.4)/360
d = 2*50*np.tan(math.radians(0.5*FOV)) * 1/deg2km
d = d/2

#############################
#DATI REALI
#Lat, Long, Alt
df = pd.read_csv("orbite/Orbit1_nadir/Orbit1_nadir/lla2.csv") 
real_Latitudes, real_Longitudes, real_Altitudes = df['Lat (deg)'].to_numpy().astype(float), df['Lon (deg)'].to_numpy().astype(float), df['Alt (km)'].to_numpy().astype(float)

#posizione
dpf = pd.read_csv("orbite/Orbit1_nadir/Orbit1_nadir/pos2.csv") 
real_X, real_Y, real_Z  = dpf['x (km)'].to_numpy().astype(float), dpf['y (km)'].to_numpy().astype(float),dpf['z (km)'].to_numpy().astype(float)

#angoli di Eulero
dq = pd.read_csv("orbite/Orbit1_nadir/Orbit1_nadir/att2.csv")  
real_t1 = dq['yaw (deg)'].to_numpy().astype(float)
real_t2 = dq['pitch (deg)'].to_numpy().astype(float)  
real_t3 = dq['roll (deg)'].to_numpy().astype(float)

# #velocità angolari
real_om1, real_om2, real_om3 = [(float(real_t1[i+1])-float(real_t1[i]))/10 for i in range(len(dq)-2)],[(float(real_t2[i+1])-float(real_t2[i]))/10 for i in range(len(dq)-2)],[(float(real_t3[i+1])-float(real_t3[i]))/10 for i in range(len(dq)-2)]
real_om1, real_om2, real_om3 = np.array(real_om1), np.array(real_om2), np.array(real_om3)
# #database crateri
DB = pd.read_csv(r"/Users/robertodelprete/Documents/PyScripts/GitRepo/KalmanFilter/orbite/lunar_crater_database_robbins_2018.csv") 

real_Latitudes, real_Longitudes, real_Altitudes = real_Latitudes[SFT:], real_Longitudes[SFT:], real_Altitudes[SFT:]
real_X, real_Y, real_Z  = real_X[SFT:], real_Y[SFT:], real_Z[SFT:]
real_t1, real_t2, real_t3 = real_t1[SFT:], real_t2[SFT:], real_t3[SFT:]
real_om1, real_om2, real_om3 = real_om1[SFT:], real_om2[SFT:], real_om3[SFT:]


### CLASSE PNP

class PnPSolver:
     """
     Class for solving the PnP problem.
     """
    
     def __init__(self, SatPosition:np.array,):
          """
          Args:
               SatPosition (np.array): Satellites position expressed in LCLF frame.
          """
          self.x,self.y,self.z = SatPosition
          print(f'Satellite position (real):{self.x,self.y,self.z}')
          
          # Camera Intrinsic Parameters:
          self.PictureFormat=1024
          self.__S = 865.0945322202958 # m (focal length 600 mm per la wide angle camera)
          cx=self.PictureFormat//2
          cy=self.PictureFormat//2

          self.K_matrix=np.array([(self.__S, 0, cx),(0,self.__S,cy),(0,0,1)])  #camera matrix
          self.dist_coeffs = np.zeros((4,1))
          
          # Catalogue Search Params
          self.min_diam = 1
          self.max_diam = 30


     def update_swath(self):
          self.H, self.Lat, self.Lon = cartesian2spherical(self.x,self.y,self.z)
          deg2km=(2*math.pi*1737.4)/360 # degree to km convertion
          d = 2*self.H*np.tan(math.radians(0.5*FOV)) * 1/deg2km # Swathwidth expressed in degree
          self.__d = d/2  # Half of the distance for making the bounds (private scope)

     def find_craters(self, DB:pd.DataFrame):
          """
          Args:
               DB (pd.Dataframe): Crater catalog of Robbins et al. containg the craters in LLA.
          """
          self.update_swath()
          lat_bounds, lon_bounds = [self.Lat-self.__d, self.Lat+self.__d],[self.Lon-self.__d, self.Lon+self.__d]
          craters_cat = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME='ROBBINS')
          self.craters_cat = craters_cat[(craters_cat['Diam'] < self.max_diam)&(craters_cat['Diam'] >= self.min_diam)]
          assert len(self.craters_cat) > 5, f'The number of craters ({len(self.craters_cat)}) is unsufficient for the PnP.'

     def make3DPoints(self, inv=False):
          """
          Args:
               Creates the points in the world reference frame.
          """
          tmp = []
          for idx, row in self.craters_cat.iterrows():
               cLat, cLon, cAlt = row['Lat'], row['Lon'], 0 # crater latitude and longitude and altitude
               cx,cy,cz = spherical2cartesian(cAlt, cLat, cLon)
               tmp.append(np.array([cx,cy,cz]))
          self.Points3D = np.vstack(tmp)
               
     def make2DPoints(self, swap_x=False,swap_y=False):
          """
          Args:
               Creates the points in the camera reference frame.
          """
          self.update_swath()
          tmp = []
          topleft = np.array([self.Lat +self.__d,self.Lon-self.__d]) 
          bottleft = np.array([self.Lat -self.__d,self.Lon-self.__d]) 
          topright = np.array([self.Lat +self.__d,self.Lon+self.__d]) 

          ky = abs(topleft[0]-bottleft[0])
          kx = abs(topleft[1]-topright[1])

          
          for idx, row in self.craters_cat.iterrows():
               cLat, cLon, cAlt = row['Lat'], row['Lon'], 0 # crater latitude and longitude and altitude
               uv = np.array([(topleft[0]-cLat)*self.PictureFormat/ky, (cLon-topleft[1])*self.PictureFormat/kx])
               if swap_x:
                    uv = np.array([self.PictureFormat-float(uv[0]), float(uv[1])])
               if swap_y:
                    uv = np.array([float(uv[0]), self.PictureFormat-float(uv[1])] )

               tmp.append(uv)
          self.Points2D = np.vstack(tmp)

     def build_image(self):
          """
          Args:
               Displays an image of the craters on the lunar surface.
          """
          assert len(self.craters_cat) > 5, f'The number of craters ({len(self.craters_cat)}) is unsufficient for the PnP.'
          plt.figure(dpi=180, figsize=(6,6))
          plt.scatter([x[0] for x in self.Points2D], [x[1] for x in self.Points2D])
          plt.axis(False)
          plt.show()

     def multiSolve(self):
          """
          Args:
               Calculates multiple solutions of the PnP problem.
          """
          mulisol = cv2.solvePnPGeneric(self.Points3D, self.Points2D, self.K_matrix, self.dist_coeffs) #no n limits
          return mulisol
     
     def solvePnP(self, method):

          def check_position(position):
               h,lat,lon=cartesian2spherical(position[0],position[1],position[2])
               x,y,z = spherical2cartesian(h,lat,lon)
               return [x,y,z]

          knownMethods = [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP,cv2.SOLVEPNP_ITERATIVE]
          if type(method) == int:
               method = knownMethods[method]
          assert method in knownMethods, "Method not recognized."
          success, rotation_vector, translation_vector = cv2.solvePnP(self.Points3D, self.Points2D, self.K_matrix, self.dist_coeffs, flags=method) #no n limits
          if success:
               rmat = cv2.Rodrigues(rotation_vector)[0]
               camera_position = -rmat.T @ translation_vector
               position=camera_position.T[0]
               position = check_position(position)
               att=rotation_matrix_to_attitude_angles(rmat)
               return position,rotation_vector
          else:
               return None



###### ERROR ANALYSIS:

res = {'x':[], 'y':[], 'z':[],'rotation_vector':[]}
for idx in range(10000):
     Pos = real_X[idx], real_Y[idx], real_Z[idx]
     solver = PnPSolver(Pos)
     try:
          solver.find_craters(DB)
          solver.make3DPoints()
          solver.make2DPoints()
          pos, rotation_vector = solver.solvePnP(0)
          res['x'].append(pos[0])
          res['y'].append(pos[1])
          res['z'].append(pos[2])
          res['rotation_vector'].append(rotation_vector)
     except:
          res['x'].append(None)
          res['y'].append(None)
          res['z'].append(None)
          res['rotation_vector'].append(None)

     
     with open('test_minDiam_1_maxDiam_30.pkl', 'wb') as handle:
          pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


######### PLOT #########
Bound = .4

x_true = real_X[:len(res['x'])]
y_true = real_Y[:len(res['y'])] 
z_true = real_Z[:len(res['z'])]

lw=1

plt.figure(dpi=180, tight_layout=True, figsize=(19,7))
plt.subplot(311)
x_pred = np.array(res['x'])
x_true = np.array(x_true)
diff_x = find_diff(x_pred,x_true)
plt.title('Error along X ')
plt.ylim([-Bound,Bound])
plt.plot(diff_x, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(312)
y_pred = np.array(res['y'])
y_true = np.array(y_true)
diff_y = find_diff(y_pred , y_true)
plt.title('Error along Y ')
plt.ylim([-Bound,Bound])
plt.plot(diff_y, '-k', linewidth=lw)
plt.ylabel('Km')

plt.subplot(313)
z_pred = np.array(res['z'])
z_true = np.array(z_true)
diff_z = find_diff(z_pred,z_true)
plt.title('Error along Z ')
plt.plot(diff_z, '-k', linewidth=lw)
plt.xlabel(f'Step Size: {dt}')
plt.ylabel('Km')
plt.ylim([-.3,.3])
plt.savefig('imgs/position_errors_LCLF.png')

plt.show()