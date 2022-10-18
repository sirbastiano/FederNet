import numpy as np
from numpy.linalg import inv, pinv
import math
from poliastro.bodies import Moon 
from astropy import units as u
from poliastro.twobody.propagation import cowell as cowell
from poliastro.core.perturbations import J3_perturbation, J2_perturbation
from poliastro.core.propagation import func_twobody
import sympy as sym
from scipy.integrate import solve_ivp
import pandas as pd    
import matplotlib.pyplot as plt

#FOV=2.85 #° NARROW ANGLE CAMERA
FOV=61.4 #° WAC

# Funzione per passare da LCLF a ENU
def LCLF2ENU(x, y, z, Lat, Long):
    phi=math.radians(Lat)
    lam=math.radians(Long)
    rot_Matrix=np.array([[-np.sin(lam), np.cos(lam), 0], [-np.cos(lam)*np.sin(phi), -np.sin(lam)*np.sin(phi), np.cos(phi)],[np.cos(lam)*np.cos(phi), np.sin(lam)*np.cos(phi), np.sin(phi)]])
    LCLF=np.array([x,y,z])
    E, N, U = np.dot(rot_Matrix,LCLF)
    return np.array(E), np.array(N), np.array(U)

# Funzione per passare da ENU a LCLF
def ENU2LCLF(e, n, u, Lat, Long):
    phi=math.radians(Lat)
    lam=math.radians(Long)
    rot_Matrix=np.array([[-np.sin(lam), -np.cos(lam)*np.sin(phi), np.cos(lam)*np.cos(phi)], [np.cos(lam), -np.sin(lam)*np.sin(phi), np.sin(lam)*np.cos(phi)],[0, np.cos(phi), np.sin(phi)]])
    ENU=np.array([e,n,u])
    x, y, z = np.dot(rot_Matrix,ENU)
    return np.array(x), np.array(y), np.array(z)

def SW_nadir(H):
    SW1=2*H*np.tan(math.radians(0.5*FOV)) 
    SW=SW1*1000 #in metri
    return SW

#########################
#MATRICE PHI

def find_F_matrix(x:float,y:float,z:float,omega1:float,omega2:float,omega3:float)-> np.array:
    #Inerzia del satellite
    II1=2000.7
    II2=2019.3#kg*m^2, VALORI DA AGGIUSTARE
    II3=647.3

    Ip1=(II3-II2)/II1
    Ip2=(II1-II3)/II2
    Ip3=(II2-II1)/II3
    mi = 4.9048695e3 # km^3/s^2 
    I_3x3=np.eye(3)
    Zero_3x3=np.zeros((3,3))
    	
    r = np.sqrt( x**2 + y**2 + z**2)

    J = np.zeros((3,3))
    # First Row
    J[0,0] = ((3*mi*(x**2))/(r**5)) - mi/(r**3)
    J[0,1] = (3*mi*x*y)/(r**5)
    J[0,2] = (3*mi*x*z)/(r**5) 
    # Second Row
    J[1,0] = -(3*mi*y*x)/(r**5)
    J[1,1] = -((3*mi*(y**2))/(r**5)) + mi/(r**3) 
    J[1,2] = -(3*mi*y*z)/(r**5) 
    # Third Row
    J[2,0] = -(3*mi*z*x)/(r**5)
    J[2,1] = -(3*mi*z*y)/(r**5) 
    J[2,2] = -((3*mi*(z**2))/(r**5)) + mi/(r**3) 
    # End


    dfomegadomega = np.zeros((3,3))

    dfomegadomega[0,0]= 0
    dfomegadomega[0,1]= -omega3*Ip1
    dfomegadomega[0,2]= -omega2*Ip1 

    dfomegadomega[1,0]= -omega3*Ip2
    dfomegadomega[1,1]= 0
    dfomegadomega[1,2]= -omega1*Ip2

    dfomegadomega[2,0]= -omega2*Ip3
    dfomegadomega[2,1]= -omega1*Ip3
    dfomegadomega[2,2]= 0
 
    # Bulk
    tmp1 = np.hstack((Zero_3x3,I_3x3,Zero_3x3,Zero_3x3)) 
    tmp2 = np.hstack((J,Zero_3x3,Zero_3x3,Zero_3x3))
    tmp3 = np.hstack((Zero_3x3,Zero_3x3,Zero_3x3,I_3x3))
    tmp4 = np.hstack((Zero_3x3,Zero_3x3,Zero_3x3,dfomegadomega))

    phi = np.vstack((tmp1,tmp2,tmp3,tmp4))

    return phi

    
###################
#FILTRO DI KALMAN

#X: The mean state estimate of the previous step (k-1). 
#P: The state covariance of previous step (k−1). 
#A: The transition nxn matrix. 
#Q: The process noise covariance matrix. 
"""
def kf_predict(X, P, A, Q):     
    X = np.dot(A, X)      
    P = np.dot(A, np.dot(P, A.T)) + Q     
    return(X,P)
"""
def kf_predict(P, A, Q):           
    P = np.dot(A, np.dot(P, A.T)) + Q     
    return(P)

#At the time step k, the update step computes the posterior mean X and covariance P of the system state given a new measurement Y. 
# The Python function kf_update performs  the update  of X  and P  giving  the predicted X  and P  matrices, the measurement vector Y, 
# the measurement matrix H and the measurement covariance matrix R. 
# The additional input will be:  
# K: the Kalman Gain matrix 


def kf_update(X, P, Y, H, R):     
    IM = np.dot(H, X)  
    PHT = np.dot(P, H.T)
    S = np.dot(H, PHT) + R
    IS = inv(S)
    K = np.dot(PHT, IS)     
    X = X + np.dot(K, (Y-IM))     
    P = P - np.dot(K, np.dot(IS, K.T))         
    return (X,P,K,IM,IS)


#PERTURBATIONS FOR COWELL PROPAGATION
def f(t0, u_, k):
    du_kep = func_twobody(t0, u_, k)
    ax, ay, az = J2_perturbation(
    t0, u_, k, J2=Moon.J2.value, R=Moon.R.to(u.km).value)
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad



def attitude_propagator(yaw0,pitch0,roll0, yaw_rate0, pitch_rate0, roll_rate0, time_span):

    II1=2000.7
    II2=2019.3#kg*m^2, VALORI DA AGGIUSTARE
    II3=647.3

    y0 = np.empty(6)

    y0[0]=yaw0

    y0[1]=pitch0

    y0[2]=roll0

    y0[3]=yaw_rate0/10

    y0[4]=pitch_rate0/10

    y0[5]=roll_rate0/10



    def dSdt(t, S):

        teta1,teta2,teta3, w1,w2,w3 = S

        tau1, tau2, tau3 = -0.00075 *t,  0.0015*t, 0.002*t

        return [w1,w2,w3, (II2-II3)/II1 * w2*w3+tau1, (II3-II1)/II2 *w1*w3+tau2, (II1-II2)/II3 *w1*w2+tau3]



    t_span = [0, time_span]

    t = np.linspace(t_span[0], time_span, int(time_span*100))

    rtol, atol = (1e-12, 1e-12)

    sol = solve_ivp(fun=dSdt, t_span=t_span, y0=y0, method='DOP853', rtol=rtol, atol=atol, t_eval=t)

    if sol.success:

        return sol.y

    else:

        return None