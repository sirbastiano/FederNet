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
# def matching(crat_det, crat_cat):
#
#    mat1 = crat_det
#    mat2 = crat_cat
#
#    MATCHER = []
#    for i in range(mat1.shape[0]):
#        for j in range(mat2.shape[0]):
#            f1 = mat1[i, :]
#            f2 = mat2[j, :]
#            tmp = sq_dif(f1, f2)
#            TMP = [i, j, tmp]
#            MATCHER.append(TMP)
#    flag = np.zeros(2)
#    for elem in MATCHER:
#        if elem[2] < 100:
#            match = [elem[0], elem[1]]
#            flag = np.vstack((flag, match))
#
#    if flag.shape != (2,):
#        flag = flag[1:, :]
#    else:
#        return None
#    flag = np.array(flag).astype(int)
#    # flag = remove_mutliple_items(flag)
#    return flag


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
            KM_to_PX = 1/0.118
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
    crater_catalogued_onboard = craters_to_relative_frame(
        df, lon_bounds, lat_bounds)
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
    filepath = "/home/sirbastiano/Desktop/Python Projects/Progetto Tesi/DATA/lunar_crater_database_robbins_2018.csv"
    DB = pd.read_csv(filepath, sep=",")
    df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME="ROBBINS")

    crater_catalogued_onboard = craters_to_relative_frame(
        df, lon_bounds, lat_bounds)
    # MATCHING
    indexes = matching(crat_det=craters_detected,
                       crat_cat=crater_catalogued_onboard)
    if indexes is not None:
        FEATURE = np.zeros(3)
        for i in range(indexes.shape[0]):
            f = df.iloc[indexes[i][1]]
            H, Lat, Lon = 0, f.Lat, f.Lon
            x_f, y_f, z_f = spherical2cartesian(H, Lat, Lon)
            FEATURE = np.vstack([FEATURE, [x_f, y_f, z_f]])
        FEATURE = FEATURE[1:, :]
        return FEATURE, indexes
    else:
        return None


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


def sort_mat(mat: np.array):
    mat_df = pd.DataFrame(mat, columns=["x", "y", "r"])
    mat_sort_x = mat_df.sort_values(by=["x"]).copy()
    mat_sort_x.insert(3, "i", range(len(mat_sort_x)))
    mat_sort_xy = mat_sort_x.sort_values(by=["y"])
    mat_sort_xy.insert(4, "j", range(len(mat_sort_xy)))
    return mat_sort_xy.sort_index()


def compute_K_vet(triplet):
    a, b, c = compute_sides(triplet)
    A, B, C = findAngles(a, b, c)
    K_vet = np.array([A, B, C])
    if K_vet is not None:
        return K_vet


def compute_sides(triplet):
    a = np.linalg.norm(triplet[0][0:2] - triplet[1][0:2])
    b = np.linalg.norm(triplet[1][0:2] - triplet[2][0:2])
    c = np.linalg.norm(triplet[2][0:2] - triplet[0][0:2])
    return a, b, c


def find_other_triplet(triplet, STORED, PICKS, HP):
    pick1 = triplet[0]
    pick2 = triplet[1]
    pick3 = triplet[2]

    Names = [pick1.name, pick2.name, pick3.name]
    names = set(Names)

    cond1 = (names in STORED)
    cond2 = ((pick2.name == pick3.name) | (
        pick3.name == pick1.name) | (pick1.name == pick2.name))

    jj = 3
    while (cond1 & cond2):
        try:
            pick2 = PICKS.iloc[int(HP[jj-1, 0])]
            pick3 = PICKS.iloc[int(HP[jj, 0])]
            Names = [pick1.name, pick2.name, pick3.name]
            names = set(Names)
            jj += 1
        except IndexError:
            return None

    return [pick1, pick2, pick3]


def find_triplet(df, idx_a: int, idx_b: int):
    if idx_a == idx_b:
        print('Index must be different!')
        pass
    # INPUT: df sorted ij, index
    # OUTPUT: triplet

    def ij_picks(pick, df, n):
        i = pick.i
        j = pick.j
        for k in range(n):
            if k == 0:
                PICKS = df[
                    (df.j == j - k)
                    | (df.j == j + k)
                    | (df.i == i + k)
                    | (df.i == i - k)
                ]
            else:
                tmp = df[
                    (df.j == j - k)
                    | (df.j == j + k)
                    | (df.i == i + k)
                    | (df.i == i - k)
                ]
                PICKS = pd.concat([PICKS, tmp])
        return PICKS

    pick1 = df.iloc[idx_a]
    pick2 = df.iloc[idx_b]

    n = 25

    PICKS = ij_picks(pick1, df, n)
    PICKS = PICKS.drop_duplicates()

    ind = 0
    HP = np.zeros(2)  # Hypothesis
    for pick in PICKS.iloc:
        if pick.name != pick1.name:
            dist_1 = np.linalg.norm(pick[0:2] - pick1[0:2])
            dist_2 = np.linalg.norm(pick[0:2] - pick2[0:2])
            dist_12 = min([dist_1, dist_2])
            hp = np.hstack([ind, dist_12])
            HP = np.vstack([HP, hp])
        ind += 1

    if HP.shape[0] > 3:  # At least 3 craters
        HP = HP[1:, :].copy()  # Remove first zeros
        HP.view("f8,f8").sort(order=["f1"], axis=0)  # Order by Eu Distance
        pick3 = df.iloc[int(HP[0, 0])]
        if pick2.name == pick3.name:
            jj = 1
            while (pick2.name == pick3.name) | (pick1.name == pick3.name):
                try:
                    pick3 = PICKS.iloc[int(HP[jj, 0])]
                    jj += 1
                except IndexError:
                    return None

        return [pick1, pick2, pick3], HP, PICKS
    else:
        return None


def picktrip(TRI, idx):
    triplet, IDs = TRI[idx], []

    for index in triplet:
        IDs.append(index)

    crat1 = craters_det_sort.iloc[IDs[0]]
    crat2 = craters_det_sort.iloc[IDs[1]]
    crat3 = craters_det_sort.iloc[IDs[2]]
    return crat1, crat2, crat3


def findAngles(a, b, c):
    # applied cosine rule
    A = np.arccos((b * b + c * c - a * a) / (2 * b * c))
    B = np.arccos((a * a + c * c - b * b) / (2 * a * c))
    C = np.arccos((b * b + a * a - c * c) / (2 * b * a))
    # convert into degrees
    A, B, C = np.rad2deg(A), np.rad2deg(B), np.rad2deg(C)
    return A, B, C


def dist_ctrs(c1, c2, input='deg'):
    x = c1[0:2]
    y = c2[0:2]
    if input == 'deg':
        deg2km = 2 * np.pi * 1737.4 / 360
        return eu_dist(x, y) * deg2km
    else:
        return eu_dist(x, y)


def find_max_crt_dist(triplet):
    a = dist_ctrs(triplet[0], triplet[1])
    b = dist_ctrs(triplet[1], triplet[2])
    c = dist_ctrs(triplet[2], triplet[0])
    return np.max([a, b, c])


def find_max_radius(triplet):
    a = triplet[0].r
    b = triplet[1].r
    c = triplet[2].r
    return np.max([a, b, c])  # Expressed in Km


def find_bounds(triplet):
    a = triplet[0]
    b = triplet[1]
    c = triplet[2]
    lat_min = np.min([a.lat, b.lat, c.lat])
    lat_max = np.max([a.lat, b.lat, c.lat])
    lon_min = np.min([a.lon, b.lon, c.lon])
    lon_max = np.max([a.lon, b.lon, c.lon])
    return lat_min, lat_max, lon_min, lon_max


def main():
    pass


if __name__ == "__main__":
    main()
