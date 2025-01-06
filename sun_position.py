from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy import units as u


def sun_position(epoch):
    """计算给定历元的太阳系质心相对地球的位置

    Args:
        epoch (Any): 历元

    Returns:
        tuple: 归一化的笛卡尔坐标[x, y, z]
    """
    t = Time(epoch, format='jyear')
    with solar_system_ephemeris.set('de440'):
        earth = get_body_barycentric('earth', t)
        x_au = earth.x.to(u.au)
        y_au = earth.y.to(u.au)
        z_au = earth.z.to(u.au)
        return -x_au.value, -y_au.value, -z_au.value
