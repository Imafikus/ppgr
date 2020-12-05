import main as ppgr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


def lerp(q1, q2, t_u, t):
    q = q1 * (1 - (t / t_u)) + q2 * (t / t_u)

def slerp(q1, q2, t_u, t):
    cos_theta = np.dot(q1, q2)
    if cos_theta < 0:
        q1 = -q1
        cos_theta = -cos_theta
    
    if cos_theta > 0.95:
        return lerp(q1, q2, t_u, t)
    
    phi_0 = np.arccos(cos_theta)

    q = q1 * (np.sin(phi_0 * 1 - t / t_u) / np.sin(phi_0)) + q2 * (np.sin(phi_0 * t / t_u) / np.sin(phi_0))
    return q

def transform(v, q):
	p, phi = ppgr.Q2AngleAxis(q)
	Rp = ppgr.Rodrigez(p, phi)
	res = np.matmul(Rp, v)
	return res

if __name__ == "__main__":
    frames = 60

    # temena pocetne i kranje pozicije
    c_start = np.array([1, 1, 1])
    c_end = np.array([3, 5, 7])
    
    # Pocetna duzina stranica
    startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    # pocetne rotacije
    q0 = np.array([1.0 ,3.0, 3.0 , 1.0])
    q1 = np.array([10.0, 1.0, 2.0, 3.0])


    # inicijalizacija animacije i iscrtavanja
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim((0, 8))
    ax.set_ylim((0, 8))
    ax.set_zlim((0, 8))

    colors = ['r', 'g', 'b']

    # iscrtavanje pocetne i kranje pozicije
    for i in range(3):
        # iscrtaj pocetnu poziciju
        start = transform(startpoints[i], q0).flatten() + c_start
        start = start.tolist()[0]

        end = transform(endpoints[i], q0) + c_start 
        end = end.tolist()[0]

        ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])

        #iscrtaj krajnju poziciju
        start = transform(startpoints[i], q1) + c_end
        start = start.tolist()[0]

        print('start: ', start)

        end = transform(endpoints[i], q1) + c_end
        end = end.tolist()[0]

        ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])


    plt.show()
