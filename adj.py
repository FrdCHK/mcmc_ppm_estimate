"""
平差拟合五参数
@Author: Jingdong Zhang
@DATE  : 2022/4/27
"""
import numpy as np

from sun_position import sun_position
from units import mas2rad, rad2mas, mas2deg


def adj(data, cov, ref_epo=None):
    dec = np.deg2rad(data['DEC'])
    mean_dec = np.mean(dec)
    sDEC = np.sin(mean_dec)
    cDEC = np.cos(mean_dec)
    dy = rad2mas(dec - mean_dec)

    ra = np.deg2rad(data['RA']) * cDEC
    mean_ra = np.mean(ra)
    dx = rad2mas(ra - mean_ra)
    delta_t = data.at[data.index.size-1, "EPOCH"] - data.at[0, "EPOCH"]
    if ref_epo is None:
        t_0 = (data.at[data.index.size-1, "EPOCH"] + data.at[0, "EPOCH"]) / 2
    else:
        t_0 = ref_epo
    X_0 = np.array([[0], [0], [mas2rad(5)], [(dx.iat[-1] - dx.iat[0]) / delta_t], [(dy.iat[-1] - dy.iat[0]) / delta_t]])
    sRA = np.sin(mean_ra/cDEC)
    cRA = np.cos(mean_ra/cDEC)

    B = np.zeros([2 * data.index.size, 5])
    Q = np.zeros([2 * data.index.size, 2 * data.index.size])
    L = np.zeros([2 * data.index.size, 1])
    for i, item in data.iterrows():
        L[2 * i:2 * i + 2] = np.array([[dx[i]], [dy[i]]])
        sun_x, sun_y, sun_z = sun_position(item['EPOCH'])
        B[2 * i:2 * i + 2] = np.array([[1, 0, sun_y * cRA - sun_x * sRA, item['EPOCH'] - t_0, 0],
                                       [0, 1, sun_z * cDEC - sun_x * cRA * sDEC - sun_y * sRA * sDEC, 0,
                                        item['EPOCH'] - t_0]])
        Q[2 * i:2 * i + 2, 2 * i:2 * i + 2] = cov[i]
    l = L - B @ X_0
    P = np.linalg.inv(Q)
    N_BB = B.T @ P @ B
    W = B.T @ P @ l
    Q_xx = np.linalg.inv(N_BB)
    x = Q_xx @ W
    res = X_0 + x
    res[0] = (np.rad2deg(mean_ra) + mas2deg(res[0])) / cDEC
    res[1] = np.rad2deg(mean_dec) + mas2deg(res[1])
    return res, Q_xx

# def adj(data, cov):
#     ra = np.deg2rad(data['RA']) * np.cos(np.deg2rad(np.mean(data['DEC'])))
#     dec = np.deg2rad(data['DEC'])
#     delta_t = data.at[data.index.size-1, "EPOCH"] - data.at[0, "EPOCH"]
#     t_0 = (data.at[data.index.size-1, "EPOCH"] + data.at[0, "EPOCH"]) / 2
#     X_0 = np.array([[np.mean(ra)], [np.mean(dec)], [mas2rad(10.0)],
#                     [(ra.iat[-1] - ra.iat[0]) / delta_t],
#                     [(dec.iat[-1] - dec.iat[0]) / delta_t]])
#     sRA = np.sin(X_0[0, 0]/np.cos(X_0[1, 0]))
#     cRA = np.cos(X_0[0, 0]/np.cos(X_0[1, 0]))
#     sDEC = np.sin(X_0[1, 0])
#     cDEC = np.cos(X_0[1, 0])
#
#     B = np.zeros([2 * data.index.size, 5])
#     Q = np.zeros([2 * data.index.size, 2 * data.index.size])
#     L = np.zeros([2 * data.index.size, 1])
#     for i, item in data.iterrows():
#         L[2 * i:2 * i + 2] = np.array([[ra[i]], [dec[i]]])
#         sun_x, sun_y, sun_z = sun_position(item['EPOCH'])
#         B[2 * i:2 * i + 2] = np.array([[1, 0, sun_y * cRA - sun_x * sRA, item['EPOCH'] - t_0, 0],
#                                        [0, 1, sun_z * cDEC - sun_x * cRA * sDEC - sun_y * sRA * sDEC, 0,
#                                         item['EPOCH'] - t_0]])
#         Q[2 * i:2 * i + 2, 2 * i:2 * i + 2] = cov[i]
#     l = L - B @ X_0
#     P = np.linalg.inv(Q)
#     N_BB = B.T @ P @ B
#     W = B.T @ P @ l
#     Q_xx = np.linalg.inv(N_BB)
#     x = Q_xx @ W
#     res = X_0 + x
#     res[0] = np.rad2deg(res[0]) / cDEC
#     res[1] = np.rad2deg(res[1])
#     res[2:] = rad2mas(res[2:])
#     return res, Q_xx
