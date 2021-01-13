import numpy as np
from utility.utils import *
import matplotlib.pyplot as plt
from math import sqrt

from Match.icp import icp


def sq_dif(f1, f2):
    a = abs(np.power(f1, 2) - np.power(f2, 2))
    a = np.sqrt(a)
    return np.sum(a)


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


def crater_catalogued(current_pos, catalog='ROBBINS'):
    # CATALOG SEARCH:
    x, y, z = current_pos[0], current_pos[1], current_pos[2]
    H, Lat, Lon = cartesian2spherical(x, y, z)  # x,y,z --> H, Lat, Lon
    dteta = find_dteta(H)  # d/R, where d^2 is the footprint
    sp = dteta / 2  # span
    lat_bounds, lon_bounds = (
        np.array([Lat - sp, Lat + sp]),
        np.array([Lon - sp, Lon + sp]),
    )
    if catalog == 'ROBBINS':
        filepath = "DATA/lunar_crater_database_robbins_2018.csv"
        DB = pd.read_csv(filepath, sep=",")
        df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME="ROBBINS")
    elif catalog == 'COMBINED':
        filepath = "DATA/H_L_combined.csv"
        DB = pd.read_csv(filepath, sep=",")
        df = CatalogSearch(DB, lat_bounds, lon_bounds, CAT_NAME="COMBINED")
    else:
        print("Catalog Not Available!")
        return None
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


def compute_centroid(trip):
    x1, x2, x3 = trip[0][0], trip[1][0], trip[2][0]
    y1, y2, y3 = trip[0][1], trip[1][1], trip[2][1]

    xc = (x1+x2+x3)/3
    yc = (y1+y2+y3)/3
    return [xc, yc]


def find_triplets(craters):

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
    ender = N*N
    K = np.zeros((ender, 15))
    lister = 0
    for i in range(N):
        printProgressBar(i+1, N, printEnd='')
        for j in range(N):
            MIN = np.array([9999, 9999])
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

                    D1 = eu_dist(P1, P3)
                    D2 = eu_dist(P2, P3)

                    if (D1 < MIN[0]) & (D2 < MIN[1]):
                        MIN[0], MIN[1] = D1, D2

                        d1, d2, d3 = eu_dist(P1, C), eu_dist(
                            P2, C), eu_dist(P3, C)
                        d_i, d_j, d_k = d1/r1, d2/r2, d3/r3

                        try:
                            K_v = compute_K_vet(triplet)
                            K[lister] = Hstack(
                                K_v, d_i, d_j, d_k, x1, y1, r1, x2, y2, r2, x3, y3, r3)
                        except ZeroDivisionError:
                            pass

            lister += 1

    return K[np.all(K != 0, axis=1)]


def dropduplicates(df):
    df = df.drop_duplicates(subset=['Angle1'], keep='first')
    df = df.drop_duplicates(subset=['Angle2'], keep='first')
    df = df.drop_duplicates(subset=['Angle3'], keep='first')
    return df


def inner_join(q1, q2, tol1):
    DFs, items = [], []
    s = pd.DataFrame()
    for ind, row in q1.iterrows():

        cond1 = ((abs(row.Angle1 - q2.Angle1) < tol1))
        cond1_a = ((abs(row.Angle2 - q2.Angle2) < tol1))
        cond1_b = ((abs(row.Angle3 - q2.Angle3) < tol1))
        cond1_c = ((abs(row.Angle2 - q2.Angle3) < tol1))
        cond1_d = ((abs(row.Angle3 - q2.Angle2) < tol1))
        A = ((cond1 & cond1_a & cond1_b) | (cond1 & cond1_c & cond1_d))
        ###############################################
        cond2 = ((abs(row.Angle1 - q2.Angle2) < tol1))
        cond2_a = ((abs(row.Angle2 - q2.Angle1) < tol1))
        cond2_b = ((abs(row.Angle3 - q2.Angle3) < tol1))
        cond2_c = ((abs(row.Angle2 - q2.Angle3) < tol1))
        cond2_d = ((abs(row.Angle3 - q2.Angle1) < tol1))
        B = ((cond2 & cond2_a & cond2_b) | (cond2 & cond2_c & cond2_d))
        ###############################################
        cond3 = ((abs(row.Angle1 - q2.Angle3) < tol1))
        cond3_a = ((abs(row.Angle2 - q2.Angle1) < tol1))
        cond3_b = ((abs(row.Angle3 - q2.Angle2) < tol1))
        cond3_c = ((abs(row.Angle2 - q2.Angle2) < tol1))
        cond3_d = ((abs(row.Angle3 - q2.Angle1) < tol1))
        C = ((cond3 & cond3_a & cond3_b) | (cond3 & cond3_c & cond3_d))

        s = q2[A | B | C]

        if s.shape[0] > 0:
            DFs.append(s)
            items.append(row)

    return DFs, items

# Computing centroid local:


def compute_pos(A, B):
    hp = A
    x1, x2, x3 = hp.x1, hp.x2, hp.x3
    y1, y2, y3 = hp.y1, hp.y2, hp.y3
    xc = float((x1+x2+x3)/3)
    yc = float((y1+y2+y3)/3)

    XC, YC = 850/2, 850/2

    delta_x = XC-xc
    delta_y = YC-yc
    # Recompute centroid in LAT-LON:
    hp = B
    x1, x2, x3 = hp.lon1, hp.lon2, hp.lon3
    y1, y2, y3 = hp.lat1, hp.lat2, hp.lat3
    xc_lon = (x1+x2+x3)/3
    yc_lat = (y1+y2+y3)/3

    u = 257.52  # ? DEG TO PXS
    px2deg = 1/u  # pixel to degree
    delta_x *= px2deg
    delta_y *= px2deg

    C = [xc_lon, yc_lat]
    pos = [C[0]-delta_x, C[1]-delta_y]
    return pos

def check_sol(I,J, tol, mode, S, iss):
    if mode == 'natural':
        row1 = iss[I]
        tmp = S[I].iloc[J]
    elif mode == 'inverse':
        row1 = S[I].iloc[J]
        tmp = iss[I]
    
    IDS = [0,1,2]

    left_id = np.argmin([row1.lon1, row1.lon2, row1.lon3])
    right_id = np.argmax([row1.lon1, row1.lon2, row1.lon3])
    for epsi in IDS:
        if (epsi != left_id) & (epsi != right_id):
            center_id = epsi

    if left_id==0:
        left = [row1.lon1, row1.lat1, row1.r1]
    elif left_id==1:
        left = [row1.lon2, row1.lat2, row1.r2]
    elif left_id==2:
        left = [row1.lon3, row1.lat3, row1.r3]    

    if right_id==0:
        right = [row1.lon1, row1.lat1, row1.r1]
    elif right_id==1:
        right = [row1.lon2, row1.lat2, row1.r2]
    elif right_id==2:
        right = [row1.lon3, row1.lat3, row1.r3] 
    
    if center_id==0:
        center = [row1.lon1, row1.lat1, row1.r1]
    elif center_id==1:
        center = [row1.lon2, row1.lat2, row1.r2]
    elif center_id==2:
        center = [row1.lon3, row1.lat3, row1.r3] 


    x1,x2,x3 = tmp.x1, tmp.x2, tmp.x3
    y1,y2,y3 = tmp.y1, tmp.y2, tmp.y3
    r1,r2,r3 = tmp.r1, tmp.r2,  tmp.r3

    Left_id = np.argmin([x1,x2,x3])
    Right_id = np.argmax([x1,x2,x3])
    for epsi in IDS:
        if (epsi != Left_id) & (epsi != Right_id):
            Center_id = epsi

    if Left_id==0:
        Left = [x1, y1, r1]
    elif Left_id==1:
        Left = [x2,y2,r2]
    elif Left_id==2:
        Left = [x3,y3,r3]    

    if Right_id==0:
        Right = [x1,y1,r1]
    elif Right_id==1:
        Right = [x2,y2,r2]
    elif Right_id==2:
        Right = [x3,y3,r3]

    if Center_id==0:
        Center = [x1, y1, r1]
    elif Center_id==1:
        Center = [x2,y2,r2]
    elif Center_id==2:
        Center = [x3,y3,r3] 

    a=left[2]/Left[2]
    b=right[2]/Right[2]
    c=center[2]/Center[2]

    if (abs(a-b) < tol) & (abs(a-c) < tol) & (abs(b-c) < tol):
        return True
    else: return False


def plot_sol(I,J, mode, S, iss, lon_bounds, lat_bounds, filename):
    if mode == 'natural':
        row1 = iss[I]
        tmp = S[I].iloc[J]
    elif mode == 'inverse':
        tmp = iss[I]
        row1 = S[I].iloc[J]
    
    CAMx, CAMy = ((lon_bounds[0] + lon_bounds[1]) / 2,
                  (lat_bounds[0] + lat_bounds[1]) / 2)
    
    
    crt1 = np.array([ row1.lon1, row1.lat1, row1.r1  ])
    crt2 = np.array([ row1.lon2, row1.lat2, row1.r2  ])
    crt3 = np.array([ row1.lon3, row1.lat3, row1.r3  ])
    triplet = [crt1, crt2, crt3]
    
    
    # img=cv2.imread(filename)
    img=np.zeros((850,850,3))
    deg2px = 256
    for crt in triplet:
        # crater center:
        xc, yc, rc = crt[0], crt[1], crt[2]  # This is in the absolute frame
        # f: Absolute --> f: Relative
        xc = xc - CAMx
        yc = yc - CAMy
        # f: relative --> f: OPENCV
        xc *= deg2px  # Now is in pixel not in lon deg
        yc *= deg2px  # Now is in pixel not in lat deg
        # rc *= u  # Now is in pixel not in lat deg
        
    
        xc = 850/2 + xc
        yc = 850/2 - yc
        center_coordinates = (int(xc), int(yc))
        # ? 1 km = 8.4746 px in our DEM := Merge LOLA - KAGUYA
        radius = int(crt[2] * km2px)
        color = (255, 255, 255)
        thickness = 3
        img_prova = cv2.circle(img, center_coordinates, radius, color, thickness)
    
    plt.figure(dpi=200, figsize=(10,5))
    plt.subplot(121)
    plt.imshow(img_prova)
    plt.xticks([0,848/2,848],[f'{lon_bounds[0]:.2f}°',f'{(lon_bounds[1]+lon_bounds[0])/2:.2f}°',f'{lon_bounds[1]:.2f}°'])
    plt.yticks([0,848/2,848],[f'{lat_bounds[0]:.2f}°',f'{(lat_bounds[1]+lat_bounds[0])/2:.2f}°',f'{lat_bounds[1]:.2f}°'])
    plt.xlabel('LON')
    plt.ylabel('LAT')

    # plt.xlabel('CAT')
    plt.show()
    
    
    cp1 = cv2.imread(filename)
    x1,x2,x3 = tmp.x1, tmp.x2, tmp.x3
    y1,y2,y3 = tmp.y1, tmp.y2, tmp.y3
    r1,r2,r3 = tmp.r1, tmp.r2,  tmp.r3
    cr1 = np.array([x1,y1,r1]) 
    cr2 = np.array([x2,y2,r2]) 
    cr3 = np.array([x3,y3,r3])
    crts = np.vstack([cr1,cr2,cr3])
    plt.subplot(122)
    # plt.xlabel('DET')
    plt.xticks([0,848/2,848],[f'{lon_bounds[0]:.2f}°',f'{(lon_bounds[1]+lon_bounds[0])/2:.2f}°',f'{lon_bounds[1]:.2f}°'])
    plt.yticks([0,848/2,848],[f'{lat_bounds[0]:.2f}°',f'{(lat_bounds[1]+lat_bounds[0])/2:.2f}°',f'{lat_bounds[1]:.2f}°'])
    plt.xlabel('LON')
    plt.ylabel('LAT')

    IMG1 =  img_plus_crts(cp1, crts, color="red")
    plt.imshow(IMG1)
    plt.show()

def find_slope(P1:np.array,P2:np.array) -> float:
    slope = (P2[1]-P1[1])/(P2[0]-P1[0])
    return slope


def check_sol2(I,J, tol, mode, S, iss, CAMx, CAMy):    #TODO aggiungere lat e lon bounds 
    if mode == 'natural':
        B = iss[I]
        A = S[I].iloc[J]
    elif mode == 'inverse':
        B = S[I].iloc[J]
        A = iss[I]

    hp = A
    x1_a, x2_a, x3_a = float(hp.x1), float(hp.x2), float(hp.x3)
    y1_a, y2_a, y3_a = float(hp.y1), float(hp.y2), float(hp.y3)
    r1_a, r2_a, r3_a = float(hp.r1), float(hp.r2), float(hp.r3)

    A1 = np.hstack([x1_a, y1_a, r1_a])
    A2 = np.hstack([x2_a, y2_a, r2_a])
    A3 = np.hstack([x3_a, y3_a, r3_a])

    A = np.vstack([A1, A2, A3])

    hp = B
    x1_b, x2_b, x3_b = float(hp.lon1), float(hp.lon2), float(hp.lon3)
    y1_b, y2_b, y3_b = float(hp.lat1), float(hp.lat2), float(hp.lat3)
    r1_b, r2_b, r3_b = float(hp.r1), float(hp.r2), float(hp.r3)

    x1_b_r, y1_b_r, r1_b_r = absolute2relative([x1_b, y1_b, r1_b], CAMx, CAMy)
    x2_b_r, y2_b_r, r2_b_r = absolute2relative([x2_b, y2_b, r2_b], CAMx, CAMy)
    x3_b_r, y3_b_r, r3_b_r = absolute2relative([x3_b, y3_b, r3_b], CAMx, CAMy)

    B1 = np.hstack([x1_b_r, y1_b_r, r1_b_r])
    B2 = np.hstack([x2_b_r, y2_b_r, r2_b_r])
    B3 = np.hstack([x3_b_r, y3_b_r, r3_b_r])

    B = np.vstack([B1, B2, B3])

    # identifiy points A:
    x1,x2,x3 = A[0][0], A[1][0], A[2][0]
    y1,y2,y3 = A[0][1], A[1][1], A[2][1]
    r1,r2,r3 = A[0][2], A[1][2], A[2][2]
    # Pick the ids:
    Left_id = np.argmin([x1,x2,x3])
    Right_id = np.argmax([x1,x2,x3])
    for id in [0,1,2]: 
        if (id != Left_id) & (id != Right_id): Center_id = id 
    # Reassign relate to ids:
    if Left_id==0:
        Left = [x1, y1, r1]
    elif Left_id==1:
        Left = [x2,y2,r2]
    elif Left_id==2:
        Left = [x3,y3,r3]    

    if Right_id==0:
        Right = [x1,y1,r1]
    elif Right_id==1:
        Right = [x2,y2,r2]
    elif Right_id==2:
        Right = [x3,y3,r3]

    if Center_id==0:
        Center = [x1,y1,r1]
    elif Center_id==1:
        Center = [x2,y2,r2]
    elif Center_id==2:
        Center = [x3,y3,r3]
    # Calculate Orientation:
    alfa1 = find_slope(Left, Center)
    alfa2 = find_slope(Center, Right)
    alfa3 = find_slope(Left, Right)
    # print('\n')
    # print(alfa1,alfa2, alfa3)
    # identifiy points B:
    x1,x2,x3 = B[0][0], B[1][0], B[2][0]
    y1,y2,y3 = B[0][1], B[1][1], B[2][1]
    r1,r2,r3 = B[0][2], B[1][2], B[2][2]
    # Pick the ids:
    Left_id = np.argmin([x1,x2,x3])
    Right_id = np.argmax([x1,x2,x3])
    for id in [0,1,2]: 
        if (id != Left_id) & (id != Right_id): Center_id = id 
    # Reassign relate to ids:
    if Left_id==0:
        Left = [x1, y1, r1]
    elif Left_id==1:
        Left = [x2,y2,r2]
    elif Left_id==2:
        Left = [x3,y3,r3]    

    if Right_id==0:
        Right = [x1,y1,r1]
    elif Right_id==1:
        Right = [x2,y2,r2]
    elif Right_id==2:
        Right = [x3,y3,r3]

    if Center_id==0:
        Center = [x1,y1,r1]
    elif Center_id==1:
        Center = [x2,y2,r2]
    elif Center_id==2:
        Center = [x3,y3,r3]
    # Calculate Orientation:
    beta1 = find_slope(Left, Center)
    beta2 = find_slope(Center, Right)
    beta3 = find_slope(Left, Right)
    
    if (abs(alfa1-beta1) < tol) & (abs(alfa2-beta2) < tol) & (abs(alfa3-beta3) < tol): return True
    else: return False

    
def filter_quartile(Xs):
    X = pd.DataFrame(Xs)
    Q1 = X.quantile(0.40)
    Q3 = X.quantile(0.60)
    IQR = Q3 - Q1
    X = X[np.logical_not((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]
    X = X.dropna()
    return np.array(X)


def find_ABBa(i, j, mode,S,iss, CAMx, CAMy):
        if mode == 'natural':
            tc = iss[i]
            td = S[i].iloc[j]
        elif mode == 'inverse':
            td = iss[i]
            tc = S[i].iloc[j]

        hp = td
        x1_a, x2_a, x3_a = float(hp.x1), float(hp.x2), float(hp.x3)
        y1_a, y2_a, y3_a = float(hp.y1), float(hp.y2), float(hp.y3)
        r1_a, r3_a, r3_a = float(hp.r1), float(hp.r2), float(hp.r3)

        A1 = np.hstack([x1_a, y1_a])
        A2 = np.hstack([x2_a, y2_a])
        A3 = np.hstack([x3_a, y3_a])

        A = np.vstack([A1, A2, A3])

        hp = tc
        x1_b, x2_b, x3_b = float(hp.lon1), float(hp.lon2), float(hp.lon3)
        y1_b, y2_b, y3_b = float(hp.lat1), float(hp.lat2), float(hp.lat3)
        r1_b, r2_b, r3_b = float(hp.r1), float(hp.r2), float(hp.r3)
        # Attenzione qua, passo dal riferimento assoluto a quello relativo.... sarà corretto così?
        x1_b_r, y1_b_r, r1_b_r = absolute2relative([x1_b, y1_b, r1_b], CAMx, CAMy)
        x2_b_r, y2_b_r, r2_b_r = absolute2relative([x2_b, y2_b, r2_b], CAMx, CAMy)
        x3_b_r, y3_b_r, r3_b_r = absolute2relative([x3_b, y3_b, r3_b], CAMx, CAMy)

        B1 = np.hstack([x1_b_r, y1_b_r])
        B2 = np.hstack([x2_b_r, y2_b_r])
        B3 = np.hstack([x3_b_r, y3_b_r])

        B1_a = np.hstack([x1_b, y1_b])
        B2_a = np.hstack([x2_b, y2_b])
        B3_a = np.hstack([x3_b, y3_b])

        B = np.vstack([B1, B2, B3])
        B_a = np.vstack([B1_a, B2_a, B3_a])

        return A, B, B_a


def find_pairs(A,B):

    def find_left(A):
        left_id = np.argmin([A[0,0],A[1,0],A[2,0]])
        # return A[left_id]
        return left_id

    def find_right(A):
        right_id = np.argmax([A[0,0],A[1,0],A[2,0]])
        # return A[right_id]
        return right_id

    def find_center(A):
        right_id = np.argmax([A[0,0],A[1,0],A[2,0]])
        left_id = np.argmin([A[0,0],A[1,0],A[2,0]])
        for j in range(A.shape[0]):
            if (j!= right_id) & (j!= left_id):
                center_id = j
        # return A[center_id] 
        return center_id 
    
    
    l1,r1,c1 = find_left(A), find_right(A), find_center(A)
    l2,r2,c2 = find_left(B), find_right(B), find_center(B)
    
    idx1 = np.hstack([l1,l2])
    idx2 = np.hstack([c1,c2])
    idx3 = np.hstack([r1,r2])

    IDX = np.vstack([idx1,idx2,idx3])
    return IDX

def H_estimation(Is, Js, mode, S, iss, CAMx,CAMy ,VERBOSE=False):
    PX2DEGs = []
    for s in range(len(Is)):
        i, j = Is[s], Js[s]
        A,B,B_a = find_ABBa(i,j,mode, S, iss, CAMx, CAMy)
        comb=find_pairs(A,B)
        if VERBOSE:
            print(f'Pair result is:\n{comb}')
        
        ##########################################################################
        # Primo Cratere:                                                        1
        CR = 0
        i, j = int(comb[CR,0]), int(comb[CR,1])
        cr1_a = A[i]   # Relativo
        cr1_b = B_a[j] # Assoluto

        # Secondo Cratere:
        CR = 1
        y, k = int(comb[CR,0]), int(comb[CR,1])
        cr2_a = A[y]   # Relativo
        cr2_b = B_a[k] # Assoluto

        # DISTANCES:
        d_pix = eu_dist(cr1_a, cr2_a) # Distance crater1-crater2 in pixel
        d_deg = eu_dist(cr1_b, cr2_b) # Distance crater1-crater2 in degrees
        if VERBOSE:
            print(f'Distance in pix: {d_pix:.2f}\nDistance in deg: {d_deg:.2f}')
        px2deg = d_deg/d_pix
        PX2DEGs.append(px2deg)
        #########################################################################
        # Primo Cratere:                                                        2
        CR = 0
        i, j = int(comb[CR,0]), int(comb[CR,1])
        cr1_a = A[i]   # Relativo
        cr1_b = B_a[j] # Assoluto

        # Secondo Cratere:
        CR = 2
        y, k = int(comb[CR,0]), int(comb[CR,1])
        cr2_a = A[y]   # Relativo
        cr2_b = B_a[k] # Assoluto

        # DISTANCES:
        d_pix = eu_dist(cr1_a, cr2_a) # Distance crater1-crater2 in pixel
        d_deg = eu_dist(cr1_b, cr2_b) # Distance crater1-crater2 in degrees
        if VERBOSE:
            print(f'Distance in pix: {d_pix:.2f}\nDistance in deg: {d_deg:.2f}')
        px2deg = d_deg/d_pix
        PX2DEGs.append(px2deg)
        #########################################################################
        # Primo Cratere:                                                        3
        CR = 1
        i, j = int(comb[CR,0]), int(comb[CR,1])
        cr1_a = A[i]   # Relativo
        cr1_b = B_a[j] # Assoluto

        # Secondo Cratere:
        CR = 2
        y, k = int(comb[CR,0]), int(comb[CR,1])
        cr2_a = A[y]   # Relativo
        cr2_b = B_a[k] # Assoluto

        # DISTANCES:
        d_pix = eu_dist(cr1_a, cr2_a) # Distance crater1-crater2 in pixel
        d_deg = eu_dist(cr1_b, cr2_b) # Distance crater1-crater2 in degrees
        if VERBOSE:
            print(f'Distance in pix: {d_pix:.2f}\nDistance in deg: {d_deg:.2f}')
        px2deg = d_deg/d_pix
        PX2DEGs.append(px2deg)
        #######################################################################
    
    px2deg = np.mean(filter_quartile(PX2DEGs))
    ###########################################################################
    px2km_n = px2deg*deg2km         # Having this px2km_n, Estimate the height!!!!
    if VERBOSE: 
        print(f'The resulting px2km: {px2km_n:.3f}')
    # d = 2 H tg(FOV/2) -> H = d /[2 tg(FOV/2)]
    d = 849*px2km_n
    FOV = np.deg2rad(90)
    H = d / (2*np.tan(FOV/2))
    if VERBOSE:
        print(f'Height corresponding is: {H:.2f}')
    return H, px2deg

def LL_estimation(A, B, B_a, px2deg, VERBOSE=False):

    comb=find_pairs(A,B)
    pp = []
    for CR in range(3):
        i = int(comb[CR,0])
        j = int(comb[CR,1])
        C = np.array([849/2, 849/2]) # Center of image
        cr1_a = A[i]
        cr1_b = B_a[j]
        Delta = C - cr1_a
        Delta_deg = Delta*px2deg
        Pos = [cr1_b[0]+Delta_deg[0], cr1_b[1]-Delta_deg[1]]
        Pos=np.array(Pos)
        if VERBOSE:
            print(f'Position Estimated is:      LON:{Pos[0]:.3f}        LAT:{Pos[1]:.3f}')
        pp.append(Pos)
    
    lon, lat = [], []
    for p in pp:
        lon.append(p[0])
        lat.append(p[1])
    
    LON = np.mean(lon)
    LAT = np.mean(lat)
    return LON, LAT

    
def main():
    pass


if __name__ == "__main__":
    main()
