"""
由相关系数和标准差，计算协方差阵
@Author: Jingdong Zhang
@DATE  : 2022/3/28
"""
import numpy as np


def corr2cov(para):
    """由VLBI单历元的标准差构造协方差阵

    Args:
        para (Series): get_vlbi_obs()从数据库CRUD导出的数据行

    Returns:
        ndarray: 协方差阵
    """
    R = np.array([[1, 0],
                  [0, 1]])
    D = np.diag(para[['RA_ERR', 'DEC_ERR']].astype(np.float64))
    return np.dot(np.dot(D, R), D)
