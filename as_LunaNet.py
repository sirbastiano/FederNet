# LUNANET
def estimate(pos, img):
    craters_detected = detect(img)
    FEATURE = crater_match(pos, craters_detected) # Feature matched in cartesian frame
    N=FEATURE.shape[0]
    kf = KalmanFilter(dim_x = 3*N+6, dim_z = 3*N+6)




    kf.F = prompt_F(N, dt)
    kf.Q = prompt_Q(N,dt,sigma_acc=0.1, sigma_dat=1.2)
    kf.H = prompt_H(N, x_c= [x,y,z], craters_det= FEATURE) # x_c = camera
    kf.R = prompt_R(3*N+6,sigma_pix=1.2) 

# Initial State:
x_six = np.array([x, y, z, vx, vy, vz])
x = state_vector_create(x_six, FEATURE)
kf.x = np.vstack(x)
kf.u = 0
kf.P = np.eye(3*N+6) * 500.


M = int(1/dt)     # s
x,y,z=[],[],[]
Z_meas = np.ones(72)

for i in range(M):
    kf.predict()
    kf.update(Z_meas)
    x.append(kf.x[0])
    y.append(kf.x[1])
    z.append(kf.x[2])

plt.figure(dpi=150, tight_layout=True)

plt.subplot(311)
plt.title('Satellite Estimated Position')
plt.plot(x, 'k', linewidth=0.8)
plt.plot(real_X[0:len(x)], 'r', linewidth=0.8)

plt.ylabel('X')

plt.subplot(312)
plt.plot(y, 'k', linewidth=0.8)
plt.plot(real_Y[0:len(y)],'r',linewidth=0.8)
plt.ylabel('Y')

plt.subplot(313)
plt.plot(z, 'k', linewidth=0.8)
plt.plot(real_Z[0:len(z)],'r', linewidth=0.8)
plt.ylabel('Z')
plt.xlabel('Time step (0.1 sec)')

plt.show()