# from functools import partial
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation, rc
# from math import sin, cos, atan2, pi
# from IPython.display import display, Math, Latex, Markdown, HTML

import numpy as np
from sklearn.neighbors import NearestNeighbors

if __name__ == "main":
    main()


def main():
    pass


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, R, t = best_fit_transform(A, src[:m, :].T)
    return R, t
    # return T, distances, i


# def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
#    fig = plt.figure(figsize=(10, 6))
#    ax = fig.add_subplot(111)
#    ax.axis("equal")
#    if data_1 is not None:
#        x_p, y_p = data_1
#        ax.plot(
#            x_p,
#            y_p,
#            color="#336699",
#            markersize=markersize_1,
#            marker="o",
#            linestyle=":",
#            label=label_1,
#        )
#    if data_2 is not None:
#        x_q, y_q = data_2
#        ax.plot(
#            x_q,
#            y_q,
#            color="orangered",
#            markersize=markersize_2,
#            marker="o",
#            linestyle=":",
#            label=label_2,
#        )
#    ax.legend()
#    return ax
#
#
# def plot_values(values, label):
#    fig = plt.figure(figsize=(10, 4))
#    ax = fig.add_subplot(111)
#    ax.plot(values, label=label)
#    ax.legend()
#    ax.grid(True)
#    plt.show()
#
#
# def animate_results(P_values, Q, corresp_values, xlim, ylim):
#    """A function used to animate the iterative processes we use."""
#    fig = plt.figure(figsize=(10, 6))
#    anim_ax = fig.add_subplot(111)
#    anim_ax.set(xlim=xlim, ylim=ylim)
#    anim_ax.set_aspect("equal")
#    plt.close()
#    x_q, y_q = Q
#    # draw initial correspondeces
#    corresp_lines = []
#    for i, j in correspondences:
#        corresp_lines.append(anim_ax.plot([], [], "grey")[0])
#    # Prepare Q data.
#    (Q_line,) = anim_ax.plot(x_q, y_q, "o", color="orangered")
#    # prepare empty line for moved data
#    (P_line,) = anim_ax.plot([], [], "o", color="#336699")
#
#    def animate(i):
#        P_inc = P_values[i]
#        x_p, y_p = P_inc
#        P_line.set_data(x_p, y_p)
#        draw_inc_corresp(P_inc, Q, corresp_values[i])
#        return (P_line,)
#
#    def draw_inc_corresp(points_from, points_to, correspondences):
#        for corr_idx, (i, j) in enumerate(correspondences):
#            x = [points_from[0, i], points_to[0, j]]
#            y = [points_from[1, i], points_to[1, j]]
#            corresp_lines[corr_idx].set_data(x, y)
#
#    anim = animation.FuncAnimation(
#        fig, animate, frames=len(P_values), interval=500, blit=True
#    )
#    return HTML(anim.to_jshtml())
#
#
# def get_correspondence_indices(P, Q):
#    """For each point in P find closest one in Q."""
#    p_size = P.shape[1]
#    q_size = Q.shape[1]
#    correspondences = []
#    for i in range(p_size):
#        p_point = P[:, i]
#        min_dist = sys.maxsize
#        chosen_idx = -1
#        for j in range(q_size):
#            q_point = Q[:, j]
#            dist = np.linalg.norm(q_point - p_point)
#            if dist < min_dist:
#                min_dist = dist
#                chosen_idx = j
#        correspondences.append((i, chosen_idx))
#    return correspondences
#
#
# def draw_correspondeces(P, Q, correspondences, ax):
#    label_added = False
#    for i, j in correspondences:
#        x = [P[0, i], Q[0, j]]
#        y = [P[1, i], Q[1, j]]
#        if not label_added:
#            ax.plot(x, y, color="grey", label="correpondences")
#            label_added = True
#        else:
#            ax.plot(x, y, color="grey")
#    ax.legend()
#
#
# def center_data(data, exclude_indices=[]):
#    reduced_data = np.delete(data, exclude_indices, axis=1)
#    center = np.array([reduced_data.mean(axis=1)]).T
#    return center, data - center
#
#
# def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
#    cov = np.zeros((2, 2))
#    exclude_indices = []
#    for i, j in correspondences:
#        p_point = P[:, [i]]
#        q_point = Q[:, [j]]
#        weight = kernel(p_point - q_point)
#        if weight < 0.01:
#            exclude_indices.append(i)
#        cov += weight * q_point.dot(p_point.T)
#    return cov, exclude_indices
#
#
# def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
#    """Perform ICP using SVD."""
#    center_of_Q, Q_centered = center_data(Q)
#    norm_values = []
#    P_values = [P.copy()]
#    P_copy = P.copy()
#    corresp_values = []
#    exclude_indices = []
#    for i in range(iterations):
#        center_of_P, P_centered = center_data(P_copy, exclude_indices=exclude_indices)
#        correspondences = get_correspondence_indices(P_centered, Q_centered)
#        corresp_values.append(correspondences)
#        norm_values.append(np.linalg.norm(P_centered - Q_centered))
#        cov, exclude_indices = compute_cross_covariance(
#            P_centered, Q_centered, correspondences, kernel
#        )
#        U, S, V_T = np.linalg.svd(cov)
#        R = U.dot(V_T)
#        t = center_of_Q - R.dot(center_of_P)
#        P_copy = R.dot(P_copy) + t
#        P_values.append(P_copy)
#    corresp_values.append(corresp_values[-1])
#    return P_values, norm_values, corresp_values
#
#
# def dR(theta):
#    return np.array([[-sin(theta), -cos(theta)], [cos(theta), -sin(theta)]])
#
#
# def R(theta):
#    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
#
#
# def jacobian(x, p_point):
#    theta = x[2]
#    J = np.zeros((2, 3))
#    J[0:2, 0:2] = np.identity(2)
#    J[0:2, [2]] = dR(0).dot(p_point)
#    return J
#
#
# def error(x, p_point, q_point):
#    rotation = R(x[2])
#    translation = x[0:2]
#    prediction = rotation.dot(p_point) + translation
#    return prediction - q_point
#
#
# def prepare_system(x, P, Q, correspondences, kernel=lambda distance: 1.0):
#    H = np.zeros((3, 3))
#    g = np.zeros((3, 1))
#    chi = 0
#    for i, j in correspondences:
#        p_point = P[:, [i]]
#        q_point = Q[:, [j]]
#        e = error(x, p_point, q_point)
#        weight = kernel(
#            e
#        )  # Please ignore this weight until you reach the end of the notebook.
#        J = jacobian(x, p_point)
#        H += weight * J.T.dot(J)
#        g += weight * J.T.dot(e)
#        chi += e.T * e
#    return H, g, chi
#
#
# def icp_least_squares(P, Q, iterations=30, kernel=lambda distance: 1.0):
#    x = np.zeros((3, 1))
#    chi_values = []
#    x_values = [x.copy()]  # Initial value for transformation.
#    P_values = [P.copy()]
#    P_copy = P.copy()
#    corresp_values = []
#    for i in range(iterations):
#        rot = R(x[2])
#        t = x[0:2]
#        correspondences = get_correspondence_indices(P_copy, Q)
#        corresp_values.append(correspondences)
#        H, g, chi = prepare_system(x, P, Q, correspondences, kernel)
#        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
#        x += dx
#        x[2] = atan2(sin(x[2]), cos(x[2]))  # normalize angle
#        chi_values.append(chi.item(0))
#        x_values.append(x.copy())
#        rot = R(x[2])
#        t = x[0:2]
#        P_copy = rot.dot(P.copy()) + t
#        P_values.append(P_copy)
#    corresp_values.append(corresp_values[-1])
#    return P_values, chi_values, corresp_values
#
#
# def compute_normals(points, step=1):
#    normals = [np.array([[0, 0]])]
#    normals_at_points = []
#    for i in range(step, points.shape[1] - step):
#        prev_point = points[:, i - step]
#        next_point = points[:, i + step]
#        curr_point = points[:, i]
#        dx = next_point[0] - prev_point[0]
#        dy = next_point[1] - prev_point[1]
#        normal = np.array([[0, 0], [-dy, dx]])
#        normal = normal / np.linalg.norm(normal)
#        normals.append(normal[[1], :])
#        normals_at_points.append(normal + curr_point)
#    normals.append(np.array([[0, 0]]))
#    return normals, normals_at_points
#
#
# def plot_normals(normals, ax):
#    label_added = False
#    for normal in normals:
#        if not label_added:
#            ax.plot(normal[:, 0], normal[:, 1], color="grey", label="normals")
#            label_added = True
#        else:
#            ax.plot(normal[:, 0], normal[:, 1], color="grey")
#    ax.legend()
#    return ax


# from sympy import init_printing, symbols, Matrix, cos as s_cos, sin as s_sin, diff
# init_printing(use_unicode = True)
#
# def RotationMatrix(angle):
#    return Matrix([[s_cos(angle) , -s_sin(angle)], [s_sin(angle), s_cos(angle)]])
#
# def prepare_system_normals(x, P, Q, correspondences, normals):
#    H = np.zeros((3, 3))
#    g = np.zeros((3, 1))
#    chi = 0
#    for (i, j), normal in zip(correspondences, normals):
#        p_point = P[:, [i]]
#        q_point = Q[:, [j]]
#        e = normal.dot(error(x, p_point, q_point))
#        J = normal.dot(jacobian(x, p_point))
#        H += J.T.dot(J)
#        g += J.T.dot(e)
#        chi += e.T * e
#    return H, g, chi
#
# def icp_normal(P, Q, normals, iterations=20):
#    x = np.zeros((3, 1))
#    chi_values = []
#    x_values = [x.copy()]  # Initial value for transformation.
#    P_values = [P.copy()]
#    P_latest = P.copy()
#    corresp_values = []
#    for i in range(iterations):
#        rot = R(x[2])
#        t = x[0:2]
#        correspondences = get_correspondence_indices(P_latest, Q)
#        corresp_values.append(correspondences)
#        H, g, chi = prepare_system_normals(x, P, Q, correspondences, normals)
#        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
#        x += dx
#        x[2] = atan2(sin(x[2]), cos(x[2])) # normalize angle
#        chi_values.append(chi.item(0)) # add error to list of errors
#        x_values.append(x.copy())
#        rot = R(x[2])
#        t = x[0:2]
#        P_latest = rot.dot(P.copy()) + t
#        P_values.append(P_latest)
#    corresp_values.append(corresp_values[-1])
#    return P_values, chi_values, corresp_values
#
# def kernel(threshold, error):
#    if np.linalg.norm(error) < threshold:
#        return 1.0
#    return 0.0
