import main as ppgr
import numpy as np


def lerp(q1, q2, t_u, t):
    q = q1 * (1 - (t / t_u)) + q2 * (t / t_u)

def slerp(q1, q2, t_u, t):
    cos_theta = np.dot(q1, q2)
    if cos_theta < 0:
        q1 = -q1
        cos_theta = -cos_theta
    
    if cos_theta > 0.95:
        return lerp(q1, t_u, t)
    
    phi_0 = np.arccos(cos_theta)

    q = q1 * (np.sin(phi_0 * 1 - t / t_u) / np.sin(phi_0)) + q2 * (np.sin(phi_0 * t / t_u) / np.sin(phi_0))
    return q
