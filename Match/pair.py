import numpy as np
from utility.utils import *
import matplotlib.pyplot as plt
from math import sqrt

from Match.icp import icp


def sq_dif(f1, f2):
    a = abs(np.power(f1, 2) - np.power(f2, 2))
    a = np.sqrt(a)
    return np.sum(a)


# init
def matching(crat_det, crat_cat):

    mat1 = crat_det
    mat2 = crat_cat

    MATCHER = []
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[0]):
            f1 = mat1[i, :]
            f2 = mat2[j, :]
            tmp = sq_dif(f1, f2)
            TMP = [i, j, tmp]
            MATCHER.append(TMP)
    flag = np.zeros(2)
    for elem in MATCHER:
        if elem[2] < 100:
            match = [elem[0], elem[1]]
            flag = np.vstack((flag, match))
    
    if flag.shape != (2,):
        flag = flag[1:, :]
    else: return None
    flag = np.array(flag).astype(int)
    # flag = remove_mutliple_items(flag)
    return flag


def craters_to_relative_frame(df, lon_b, lat_b, u=None):
    lon_bounds = lon_b
    lat_bounds = lat_b
    # CAMERA CENTER:
    CAMx, CAMy = (
        (lon_bounds[0] + lon_bounds[1]) / 2,
        (lat_bounds[0] + lat_bounds[1]) / 2,
    )

    if u == None:  # Scale Factor
        u = 257.52  # ? DEG TO PXS
        span = (abs(lon_b[0]) - abs(lon_b[1])) * u
        span = abs(int(span))
    else:
        span = (abs(lon_b[0]) - abs(lon_b[1])) * u
        span = abs(int(span))
    # Make the img:
    # img = np.zeros((span, span), dtype=int)
    if df is None:
        print("No crater found")
        pass
    else:
        W, H = (span, span)
        # Cycle through the dataframe:
        craters = np.zeros(3)
        for i in range(df.shape[0]):
            crater = df.iloc[i]
            if crater.Diam < 100:  # Only craters < 100 km
                # crater center:
                xc, yc = crater.Lon, crater.Lat  # This is in the absolute frame
                # f: Absolute --> f: Relative
                xc = xc - CAMx
                yc = yc - CAMy
                # f: relative --> f: OPENCV
                xc *= u  # Now is in pixel not in lon deg
                yc *= u  # Now is in pixel not in lat deg
                xc = W / 2 + xc
                yc = H / 2 - yc
                # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
                KM_to_PX = 8.4746
                crater_i = [xc, yc, crater.Diam / 2 * KM_to_PX]
                craters = np.vstack([craters, crater_i])
        return craters[1:, :]


def crater_catalogued(current_pos):
    # CATALOG SEARCH:
    x, y, z = current_pos[0], current_pos[1], current_pos[2]
    H, Lat, Lon = cartesian2spherical(x, y, z)  # x,y,z --> H, Lat, Lon
    dteta = find_dteta(H)  # d/R, where d^2 is the footprint
    sp = dteta / 2  # span
    lat_bounds, lon_bounds = (
        np.array([Lat - sp, Lat + sp]),
        np.array([Lon - sp, Lon + sp]),
    )
    filepath = "DATA/lunar_crater_database_robbins_2018.csv"
    DB = pd.read_csv(filepath, sep=",")
    df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME="ROBBINS")
    crater_catalogued_onboard = craters_to_relative_frame(df, lon_bounds, lat_bounds)
    return crater_catalogued_onboard


def crater_match(current_pos, craters_detected):
    # CATALOG SEARCH:
    x, y, z = current_pos[0], current_pos[1], current_pos[2]
    H, Lat, Lon = cartesian2spherical(x, y, z)  # x,y,z --> H, Lat, Lon
    dteta = find_dteta(H)  # d/R, where d^2 is the footprint
    sp = dteta / 2  # span
    lat_bounds, lon_bounds = (
        np.array([Lat - sp, Lat + sp]),
        np.array([Lon - sp, Lon + sp]),
    )
    filepath = "DATA/lunar_crater_database_robbins_2018.csv"
    DB = pd.read_csv(filepath, sep=",")
    df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME="ROBBINS")

    crater_catalogued_onboard = craters_to_relative_frame(df, lon_bounds, lat_bounds)
    # MATCHING
    indexes = matching(crat_det=craters_detected, crat_cat=crater_catalogued_onboard)
    if indexes is not None:
        FEATURE = np.zeros(3)
        for i in range(indexes.shape[0]):
            f = df.iloc[indexes[i][1]]
            H, Lat, Lon = 0, f.Lat, f.Lon
            x_f, y_f, z_f = spherical2cartesian(H, Lat, Lon)
            FEATURE = np.vstack([FEATURE, [x_f, y_f, z_f]])
        FEATURE = FEATURE[1:, :]
        return FEATURE, indexes
    else: return None


def plt_craters(craters_detected, craters_catalogued, indexes):

    A, B = np.zeros(3), np.zeros(3)
    for i in range(len(indexes)):
        idx_a = indexes[i][0]
        idx_b = indexes[i][1]

        A = np.vstack([A, craters_detected[idx_a]])
        B = np.vstack([B, craters_catalogued[idx_b]])

    A = A[1:, :]
    B = B[1:, :]

    C, D = A[:, :2], B[:, :2]

    plt.close()
    plt.figure(1, dpi=200)
    plt.xlim([0, 850])
    plt.ylim([0, 850])

    plt.scatter(C[:, 0], C[:, 1], marker="s", s=7)
    plt.scatter(D[:, 0], D[:, 1], marker="o", s=7)
    plt.legend(["detected", "catalogued"])

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
    print(len(C), len(D))


def position_estimation(pos, craters_detected, craters_catalogued, indexes):

    x, y, z = pos[0], pos[1], pos[2]
    H, Lat, Lon = cartesian2spherical(x, y, z)

    A, B = np.zeros(3), np.zeros(3)
    for i in range(len(indexes)):
        idx_a = indexes[i][0]
        idx_b = indexes[i][1]

        A = np.vstack([A, craters_detected[idx_a]])
        B = np.vstack([B, craters_catalogued[idx_b]])

    A = A[1:, :]
    B = B[1:, :]

    C, D = A[:, :2], B[:, :2]

    R, t = icp(C, D, init_pose=None, max_iterations=20, tolerance=0.001)
    t = t * 0.118  # px to Km: 1px = 118m
    Z_m = [x, y] + t

    raddii_a = A[:, 2]
    raddii_b = B[:, 2]

    rapport = raddii_a / raddii_b
    counts, bins, bars = plt.hist(rapport, bins=300)
    plt.close()
    idx = np.argmax(counts)
    sf = bins[idx]
    # print(sf)
    new_H = H * sf
    # print(new_H)
    _, _, tmp = spherical2cartesian(new_H, Lat, Lon)

    return np.hstack([Z_m, tmp])


def main():
    pass


if __name__ == "__main__":
    main()

