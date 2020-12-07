import numpy as np
from scipy.linalg import block_diag

##


def prompt_F(N: int, dt: float):
    # Offset State Vector:
    F = np.eye(3 * N + 6)
    iX, iY, iZ, iVx, iVy, iVz = 0, 1, 2, 3, 4, 5
    F[iX, iVx] = dt
    F[iY, iVy] = dt  # F
    F[iZ, iVz] = dt
    return F


def prompt_z_f(x_f: np.array, x_c: np.array):
    # x_f: x,y,z of crater matched|cartesian
    # x_c: x,y,z of camera center|cartesian
    x_f = np.array(x_f)
    x_c = np.array(x_c)  # TODO: inserisci if
    tmp = x_f - x_c
    return tmp / np.linalg.norm(tmp)


def residual(z_f: np.array, z_f_kminus: np.array):
    z_f = np.array(z_f)  # Vector between Detected crater and camera center
    z_f_kminus = np.array(
        z_f_kminus
    )  # Vector between expected crater and camera center
    return (
        z_f - z_f_kminus
    )  # vector between crater detected & expected based on pose propagation


def prompt_H_f_i(N: int, x_f: np.array, x_c: np.array, index: int):
    x_f = np.array(x_f)
    x_c = np.array(x_c)  # TODO: inserisci if

    O3 = np.zeros([3, 3])
    I3 = np.eye(3)
    z_f = prompt_z_f(x_f, x_c)
    a = z_f * z_f
    tmp = I3 * a / np.linalg.norm(x_f - x_c)
    H_f_i = np.hstack([-tmp, O3])
    for i in range(N):
        if i == index:
            H_f_i = np.hstack([H_f_i, tmp])
        else:
            H_f_i = np.hstack([H_f_i, O3])
    return H_f_i


def prompt_H(N: int, x_c, craters_det):
    x_c = np.array(x_c)  # add h rows zeros
    for i in range(N):
        x_f = np.array(craters_det[i, :])
        H_f_i = prompt_H_f_i(N, x_f, x_c, i)
        if i == 0:
            H = H_f_i
        else:
            H = np.vstack([H, H_f_i])

    R12 = np.zeros([6, 3 * N + 6])
    H = np.vstack([R12, H])
    return H  # 3N+6 x 3N+6


def prompt_R(N, sigma_pix):
    R = np.diag([sigma_pix for i in range(N)])
    return R


def prompt_Q(N, dt, sigma_acc: float, sigma_dat: float):
    I3 = np.eye(3)
    O3 = np.zeros([3, 3])

    tmp1 = sigma_acc ** 2 * I3 * dt ** 2
    tmp2 = sigma_dat ** 2  # *I3 -->  3*N

    R1 = np.hstack([tmp1, tmp1 / 2 * dt])
    for i in range(N):
        R1 = np.hstack([R1, O3])

    R2 = np.hstack([tmp1 / 2 * dt, tmp1 / 4 * dt ** 2])
    for i in range(N):
        R2 = np.hstack([R2, O3])

    left = np.hstack([O3, O3])
    LEFT = np.vstack([left for i in range(N)])

    RIGHT = np.diag([tmp2 for i in range(3 * N)])

    SUB = np.hstack([LEFT, RIGHT])
    Q = np.vstack([R1, R2, SUB])
    return Q


def state_vector_create(x_initial: np.array, matched_features: np.array) -> np.array:

    # Function defined for create the state vector after the collection of features;
    N = matched_features.shape[0]
    x = x_initial
    for i in range(N):
        feature = matched_features[i, :]
        x = np.hstack([x, feature])
    return x


def main():
    pass


if __name__ == "__main__":
    main()

