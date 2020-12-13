from numba import njit
from astropy.coordinates.funcs import spherical_to_cartesian, cartesian_to_spherical
import numpy as np
import pandas as pd
import cv2
from copy import deepcopy

def img_plus_crts(img, craters_det, color="red"):
    # Input: Img:3 chanel, craters_det: np.array
    b = craters_det
    image = img.copy()
    for i in range(b.shape[0]):

        r = b[i][2]
        x_c, y_c = b[i][0], b[i][1]

        center_coordinates = (int(x_c), int(y_c))
        radius = int(r)
        if color == "red":
            color = (255, 0, 0)
        elif color == "green":
            color = (0, 255, 0)

        thickness = 2
        cv2.circle(image, center_coordinates, radius, color, thickness)
    return image


def eu_dist(x, y):
    x1, y1 = x[0], x[1]
    x2, y2 = y[0], y[1]
    result = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
    return result


def draw_craters(df, lon_b, lat_b, u=None):
    lon_bounds = lon_b
    lat_bounds = lat_b
    # CAMERA CENTER:
    CAMx, CAMy = (
        (lon_bounds[0] + lon_bounds[1]) / 2,
        (lat_bounds[0] + lat_bounds[1]) / 2,
    )

    if u == None:  # Scale Factor
        u = 256  # ? DEG TO PXS
        span = (abs(lon_b[0]) - abs(lon_b[1])) * 256
        span = abs(int(span))
    # Make the img:
    img = np.zeros((span, span), dtype=int)
    if df is None:
        return img
    else:
        W, H = (
            img.shape[0],
            img.shape[1],
        )  # TODO change the function to non-square shapes
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # Cycle through the dataframe:
        for i in range(df.shape[0]):
            crater = df.iloc[i]
            if crater.Diam < 100:
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
                center_coordinates = (int(xc), int(yc))
                # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
                KM_to_PX = 8.4746
                radius = int(crater.Diam / 2 * KM_to_PX)
                color = 255
                thickness = 3
                img = cv2.circle(img, center_coordinates,
                                 radius, color, thickness)
        return img


def draw_craters_on_image(df, lon_b, lat_b, img, u=None):
    if df is None:
        return img
    else:
        lon_bounds = lon_b
        lat_bounds = lat_b
        # CAMERA CENTER:
        CAMx, CAMy = (
            (lon_bounds[0] + lon_bounds[1]) / 2,
            (lat_bounds[0] + lat_bounds[1]) / 2,
        )

        if u == None:  # Scale Factor
            u = 256  # ? DEG TO PXS
            span = (abs(lon_b[0]) - abs(lon_b[1])) * 256
            span = abs(int(span))
        # Make the img:
        W, H = (
            img.shape[0],
            img.shape[1],
        )  # TODO change the function to non-square shapes
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # Cycle through the dataframe:
        for i in range(df.shape[0]):
            crater = df.iloc[i]
            if crater.Diam < 100:
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
                center_coordinates = (int(xc), int(yc))
                # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
                KM_to_PX = 8.4746
                radius = int(crater.Diam / 2 * KM_to_PX)
                color = (0, 0, 255)
                thickness = 3
                img = cv2.circle(img, center_coordinates,
                                 radius, color, thickness)
        return img


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
    return np.array(x), np.array(y), np.array(z)


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


def row(idx, df):
    return df.iloc[idx]


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def find_dteta(H) -> float:
    # Output in deg
    FOV = np.deg2rad(45)
    d = 2 * H * np.tan(FOV)
    R_m = 1737.1
    dteta = d / R_m
    dteta = np.rad2deg(dteta)
    return dteta


def remove_items(list, item):

    # using list comprehension to perform the task
    res = [i for i in list if i != item]

    return res


def remove_mutliple_items(indexes):
    # print(indexes)
    idx_a = indexes[:, 0]
    idx_b = indexes[:, 1]
    # print(idx_b)
    list_a = []
    list_b = []
    for elem_a, elem_b in zip(idx_a, idx_b):
        if (
            np.count_nonzero(idx_a == elem_a) > 1
            or np.count_nonzero(idx_b == elem_b) > 1
        ):
            list_a.append(0)
            list_b.append(0)
        else:
            list_a.append(elem_a)
            list_b.append(elem_b)
    a = remove_items(list_a, 0)
    b = remove_items(list_b, 0)
    a, b = np.vstack(a), np.vstack(b)
    v = np.hstack([a, b])
    return v


@njit
def findAngles(a, b, c):
    # applied cosine rule
    A = np.arccos((b * b + c * c - a * a) / (2 * b * c))
    B = np.arccos((a * a + c * c - b * b) / (2 * a * c))
    C = np.arccos((b * b + a * a - c * c) / (2 * b * a))
    # convert into degrees
    A, B, C = np.rad2deg(A), np.rad2deg(B), np.rad2deg(C)
    return A, B, C


@njit
def compute_K_vet(triplet):
    a, b, c = compute_sides(triplet)
    A, B, C = findAngles(a, b, c)
    K_vet = np.array([A, B, C])
    if K_vet is not None:
        return K_vet


@njit
def compute_sides(triplet):
    a = np.linalg.norm(triplet[0][0:2] - triplet[1][0:2])
    b = np.linalg.norm(triplet[1][0:2] - triplet[2][0:2])
    c = np.linalg.norm(triplet[2][0:2] - triplet[0][0:2])
    return a, b, c


def find_all_triplets(craters):

    def Hstack(K_v, i, j, k, x1, y1, r1, x2, y2, r2, x3, y3, r3):
        A = np.zeros(15)
        A[0], A[1], A[2] = K_v[0], K_v[1], K_v[2]
        A[3], A[4], A[5] = i, j, k
        A[6], A[7], A[8] = x1, y1, r1
        A[9], A[10], A[11] = x2, y2, r2
        A[12], A[13], A[14] = x3, y3, r3
        return A

    def eu_dist(x, y):
        x1, y1 = x[0], x[1]
        x2, y2 = y[0], y[1]
        result = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
        return result

    def concat(a, b, c):
        A = np.zeros((3, 3))
        A[0] = a
        A[1] = b
        A[2] = c
        return A

    # Input: np.array craters
    # Output: all triplets
    N = craters.shape[0]
    ender = N*N*N
    K = np.zeros((ender, 15))
    lister = 0
    for i in range(N):
        printProgressBar(i+1, N, printEnd='')
        for j in range(N):
            for k in range(N):
                if (i != j) & (j != k):
                    a = craters[i]
                    b = craters[j]
                    c = craters[k]
                    triplet = concat(a, b, c)
                    x1, y1, r1 = a[0], a[1], a[2]
                    x2, y2, r2 = b[0], b[1], b[2]
                    x3, y3, r3 = c[0], c[1], c[2]

                    C = np.zeros(2)  # centroid
                    C[0] = (x1+x2+x3)/3
                    C[1] = (y1+y2+y3)/3

                    P1, P2, P3 = np.zeros(2), np.zeros(2), np.zeros(2)
                    P1[0] = x1
                    P1[1] = y1
                    P2[0] = x2
                    P2[1] = y2
                    P3[0] = x3
                    P3[1] = y3

                    d1, d2, d3 = eu_dist(P1, C), eu_dist(P2, C), eu_dist(P3, C)
                    d_i, d_j, d_k = d1/r1, d2/r2, d3/r3

                    try:
                        K_v = compute_K_vet(triplet)
                        K[lister] = Hstack(
                            K_v, d_i, d_j, d_k, x1, y1, r1, x2, y2, r2, x3, y3, r3)
                    except ZeroDivisionError:
                        pass

                lister += 1
    return K[np.all(K != 0, axis=1)]


def swap_df_columns(colname_1, colname_2, df):
    tmp = deepcopy(df[colname_1])
    df[colname_1] = df[colname_2]
    df[colname_2] = tmp
    return df


global km2px, deg2km
km2px = 1/0.118
deg2km = 2*np.pi*1737.4/360


def main():
    pass


if __name__ == "__main__":
    main()
