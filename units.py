import numpy as np


def mas2rad(mas_data):
    """单位转换：毫角秒转为弧度

    Args:
        mas_data (Any): 毫角秒为单位的数据

    Returns:
        Any: 弧度为单位的数据
    """
    return np.deg2rad(mas_data/3.6e6)


def rad2mas(rad_data):
    """单位转换：弧度转为毫角秒

    Args:
        rad_data (Any): 弧度为单位的数据

    Returns:
        Any: 毫角秒为单位的数据
    """
    return np.rad2deg(rad_data)*3.6e6


def mas2deg(mas_data):
    """单位转换：毫角秒转为度

    Args:
        mas_data (Any): 毫角秒为单位的数据

    Returns:
        Any: 度为单位的数据
    """
    return mas_data/3.6e6


def deg2mas(deg_data):
    """单位转换：度转为毫角秒

    Args:
        deg_data (Any): 度为单位的数据

    Returns:
        Any: mas为单位的数据
    """
    return deg_data*3.6e6
