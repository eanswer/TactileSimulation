import numpy as np

def quat_mul(a, b):

    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.array([x, y, z, w])

    return quat

def quat_conjugate(a):
    return np.concatenate([-a[:3], a[-1:]])

def unscale(v, lower, upper):
    unscaled_v = (2.0 * v - upper - lower) / (upper - lower)
    return unscaled_v

def scale(v, lower, upper):
    scaled_v = (v + 1.) / 2. * (upper - lower) + lower
    
    return scaled_v

def warp2PI(angle):
    while angle < -np.pi:
        angle += np.pi * 2.
    while angle > np.pi:
        angle -= np.pi * 2.
    return angle
