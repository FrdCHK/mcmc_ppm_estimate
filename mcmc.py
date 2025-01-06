"""
由多历元模拟数据重新拟合五参数
@Author: Jingdong Zhang
@DATE  : 2022/4/27
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee  # 需安装tqdm显示采样进度
from multiprocessing import Pool
import corner
import copy

from corr2cov import corr2cov
from adj import adj
from sun_position import sun_position
from units import rad2mas, mas2deg

# global variables
# 使用全局变量在进程池并行时效率略高
gl_ob = None
gl_cov = None
gl_sun_pos = None
gl_trigo = None


def model(theta, delta_epo, sun_pos, trigo):
    """模型

    Args:
        theta (tuple): 参数[RA, DEC, PI, PMRA*, PMDEC]
        delta_epo (float): 相对ref_epoch的历元差
        sun_pos (tuple): 太阳的相对位置
        trigo (tuple): 三角函数值

    Returns:
        ndarray: 模型输出值[RA_m, DEC_m]
    """
    
    # 注意以下计算都是基于弧度制
    ra, dec, pi, pmra, pmdec = theta
    sun_X, sun_Y, sun_Z = sun_pos
    sRA, cRA, sDEC, cDEC = trigo
    RA_result = ra + pi * (sun_Y * cRA - sun_X * sRA) + delta_epo * pmra
    DEC_result = dec + pi * (sun_Z * cDEC - sun_X * cRA * sDEC - sun_Y * sRA * sDEC) + delta_epo * pmdec
    return np.array([RA_result, DEC_result])


def log_prior(theta, theta0):
    """对数形式先验函数，这里是均匀分布

    Args:
        theta (ndarray): 参数
        theta0 (ndarray): 初始参数

    Returns:
        float: 对数形式先验函数值
    """
    a, b, c, d, e = theta
    a0, b0, c0, d0, e0 = theta0
    if np.abs(a - a0) < 2e3 and np.abs(b - b0) < 2e3 and np.abs(c) < 1e3 and np.abs(d) < 2e3 and np.abs(e) < 2e3:
        return 0.0
    return -np.inf


def log_likelihood(theta, delta_epo):
    """对数形式似然函数

    Args:
        theta (ndarray): 参数
        delta_epo (ndarray): 自变量(相对ref_epoch的历元差)

    Returns:
        float: 对数形式似然函数值
    """
    global gl_ob, gl_cov, gl_sun_pos, gl_trigo
    num = np.shape(gl_ob)[0]
    likeli_sum = 0.
    for i in range(num):
        x = np.expand_dims(gl_ob[i, :], axis=1)  # 观测值, 列向量
        m = np.expand_dims(model(theta, delta_epo[i], (gl_sun_pos[0][i], gl_sun_pos[1][i], gl_sun_pos[2][i]), gl_trigo), axis=1)  # 模型预测值, 列向量
        likeli_sum += np.log(1/np.sqrt(np.linalg.det(gl_cov[i]))) - ((x-m).T @ np.linalg.inv(gl_cov[i]) @ (x-m))[0][0]/2
        likeli_sum += np.log(1/np.sqrt(np.linalg.det(gl_cov[i]))) - ((x-m).T @ np.linalg.inv(gl_cov[i]) @ (x-m))[0][0]/2
    return num * np.log(1 / (2 * np.pi)) + likeli_sum


def log_probability(theta, theta0, delta_epo):
    """传给emcee的函数, 先验函数与似然函数乘积的对数

    Args:
        theta (ndarray): 参数
        theta0 (ndarray): 初始参数
        delta_epo (ndarray): 自变量(相对ref_epoch的历元差)

    Returns:
        float: 先验函数与似然函数的对数和
    """
    lp = log_prior(theta, theta0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, delta_epo)


def pool_initialize(ob, cov, sun_pos, trigo):
    global gl_ob, gl_cov, gl_sun_pos, gl_trigo
    gl_ob = ob
    gl_cov = cov
    gl_sun_pos = sun_pos
    gl_trigo = trigo


def sampling(delta_epo, ob, cov, sun_pos, trigo, theta0):
    """调用emcee进行采样

    Args:
        delta_epo (ndarray): 自变量(相对ref_epoch的历元差)
        ob (ndarray): 观测量
        cov (ndarray): 观测量协方差阵
        sun_pos (tuple): 太阳位置
        trigo (tuple): 三角函数值
        theta0 (ndarray): 初始参数

    Returns:
        ndarray: 样本
    """
    pos = theta0 + 1e-10 * np.random.randn(15, 5)  # 初始参数
    nwalkers, ndim = pos.shape
    # 通过进程池实现并行
    with Pool(initializer=pool_initialize, initargs=[ob, cov, sun_pos, trigo]) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(theta0, delta_epo))
        sampler.run_mcmc(pos, 5000, progress=True)  # 运行MCMC
        return sampler.get_chain(discard=2000, flat=True)  # 舍弃收敛前的样本，可根据数据实际情况调整数量


def mcmc(data):
    print(f"{data.at[0, 'NAME']} {data.at[0, 'MODE']}")
    epoch = np.array(data['EPOCH'])
    ref_epoch = (epoch[0]+epoch[-1])/2
    delta_epoch = epoch-ref_epoch
    cov = np.zeros([data.index.size, 2, 2])
    for j, iter_item in data.iterrows():
        cov[j] = corr2cov(iter_item)
    sun_pos = sun_position(epoch)
    # 平差得到参数初值
    adj_res, adj_cov = adj(data, cov)
    adj_res = adj_res.flatten()
    print("ADJ result:")
    print(adj_res)

    dec = np.deg2rad(data['DEC'])
    mean_dec = np.mean(dec)
    sDEC = np.sin(mean_dec)
    cDEC = np.cos(mean_dec)
    dy = rad2mas(dec - mean_dec)

    ra = np.deg2rad(data['RA']) * cDEC
    mean_ra = np.mean(ra)
    dx = rad2mas(ra - mean_ra)
    obs = np.array([dx, dy]).T
    sRA = np.sin(mean_ra/cDEC)
    cRA = np.cos(mean_ra/cDEC)

    t0 = [rad2mas(np.deg2rad(adj_res[0]) * cDEC - mean_ra), rad2mas(np.deg2rad(adj_res[1]) - mean_dec), adj_res[2], adj_res[3], adj_res[4]]
    samples = sampling(delta_epoch, obs, cov, sun_pos, (sRA, cRA, sDEC, cDEC), t0)

    # 计算参数的中位数
    param_medians = np.median(samples, axis=0)
    # 计算协方差矩阵
    cov_matrix = np.cov(samples.T)
    # 提取参数的不确定性（标准差）
    param_stds = np.sqrt(np.diag(cov_matrix))

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(samples, rowvar=False)

    # 提取相关系数 (对称矩阵的上三角部分，不包括对角线)
    corr_indices = [(0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (1, 3), (1, 4),
                    (2, 3), (2, 4),
                    (3, 4)]
    correlations = [corr_matrix[i, j] for i, j in corr_indices]

    # 创建行数据
    row = [data.at[0, 'NAME'], data.at[0, 'MODE'], ref_epoch,
        (np.rad2deg(mean_ra) + mas2deg(param_medians[0]))/cDEC, param_stds[0],  # RA, RA_ERR
        np.rad2deg(mean_dec) + mas2deg(param_medians[1]), param_stds[1],  # DEC, DEC_ERR
        param_medians[2], param_stds[2],  # PLX, PLX_ERR
        param_medians[3], param_stds[3],  # PMRA, PMRA_ERR
        param_medians[4], param_stds[4],  # PMDEC, PMDEC_ERR
        *correlations  # 所有相关系数
    ]
    print("MCMC result:")
    print(row[3:13:2])
    result = pd.DataFrame([row], columns = ["NAME", "MODE", "EPOCH", "RA", "RA_ERR", "DEC", "DEC_ERR",
                                            "PLX", "PLX_ERR", "PMRA", "PMRA_ERR", "PMDEC", "PMDEC_ERR",
                                            "RA_DEC_CORR", "RA_PLX_CORR", "RA_PMRA_CORR", "RA_PMDEC_CORR",
                                            "DEC_PLX_CORR", "DEC_PMRA_CORR", "DEC_PMDEC_CORR",
                                            "PLX_PMRA_CORR", "PLX_PMDEC_CORR", "PMRA_PMDEC_CORR"])

    # 计算模型值
    model_values = np.array([model(param_medians, delta_epoch[i], (sun_pos[0][i], sun_pos[1][i], sun_pos[2][i]), (sRA, cRA, sDEC, cDEC)) for i in range(len(delta_epoch))])

    # 计算卡方
    chi_square = 0
    for i in range(len(obs)):
        diff = np.expand_dims(obs[i] - model_values[i], 1)
        chi_square += (diff.T @ np.linalg.inv(cov[i]) @ diff)[0, 0]

    result['CHI_SQUARE'] = chi_square/(len(obs)*2-5)

    # corner plot
    samples_unit_converted = copy.deepcopy(samples)
    for i in range(samples_unit_converted.shape[1]):
        if i == 1:
            samples_unit_converted[:, i] = (np.rad2deg(mean_ra) + mas2deg(samples_unit_converted[:, i]))/cDEC
        elif i == 2:
            samples_unit_converted[:, i] = np.rad2deg(mean_dec) + mas2deg(samples_unit_converted[:, i])

    labels = ["$\\alpha$", "$\\delta$", "$\\varpi$", "$\\mu_{\\alpha}$", "$\\mu_{\\delta}$"]  # 支持Latex
    fig = corner.corner(samples_unit_converted, labels=labels, show_titles=True)  # 绘图
    fig.savefig(f"image/png/{data.at[0, 'NAME']}-{data.at[0, 'MODE']}.png", bbox_inches='tight')
    fig.savefig(f"image/pdf/{data.at[0, 'NAME']}-{data.at[0, 'MODE']}.pdf", bbox_inches='tight')
    plt.close('all')

    return result
