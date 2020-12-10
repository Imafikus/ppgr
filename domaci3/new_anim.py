import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import main as ppgr

def mul_q(q1, q2):
    v1 = q1[0:3]
    w1 = q1[3]
    v2 = q2[0:3]
    w2 = q2[3]
    v = np.cross(v1, v2) + w2 * v1 + w1 * v2
    w = w1 * w2 - np.dot(v1, v2)
    return ppgr.Vector2Q(v, w)

def conjugate_q(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def inverse_q(q):
    return conjugate_q(q) / (np.linalg.norm(q)**2)

def transform(pos, q):    
    qi = inverse_q(q)

    #? Izracunaj koliko se pomerila svaka osa i vrati updateovane koordinate

    x = ppgr.Vector2Q([1, 0, 0])
    x = mul_q(mul_q(q, x), qi)
    x = np.array([[pos[i], pos[i]+x[i]] for i in range(3)])

    y = ppgr.Vector2Q([0, 1, 0])
    y = mul_q(mul_q(q, y), qi)
    y = np.array([[pos[i], pos[i]+y[i]] for i in range(3)])

    z = ppgr.Vector2Q([0, 0, 1])
    z = mul_q(mul_q(q, z), qi)
    z = np.array([[pos[i], pos[i]+z[i]] for i in range(3)])

    return (x, y, z)

def lerp(q1, q2, tm, t):
    q = (1 - (t / tm)) * q1 + (t / tm) * q2
    return q

def slerp(q1, q2, tm, t):
    cos_theta = np.dot(q1, q2)
    
    if cos_theta < 0:
        q1 = -1 * q1
        cos_theta = -cos_theta
    
    if cos_theta > 0.95:
        return lerp(q1, q2, tm, t)
    
    angle = np.arccos(cos_theta)
    q = (np.sin(angle*(1 - t / tm) / np.sin(angle))) * q1 + (np.sin(angle * (t / tm) / np.sin(angle))) * q2
    return q


if __name__ == '__main__':
    #? ukupan broj frejmova
    tm = 100
    
    #? pocetna pozicija i rotacija
    p1 = np.array([0.5, -1, 0])
    q1 = ppgr.Euler2Q(math.radians(32), math.radians(-30), math.radians(75))
    
    #? Krajnja pozicija i rotacija
    p2 = np.array([0.5, 0.9, 0.75])
    q2 = ppgr.Euler2Q(math.radians(145), math.radians(-60), math.radians(-25))

    p = p1
    q = q1

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    colors = ['r-', 'g-', 'b-']

    #? izracunaj pocetnu poziciju i iscrtaj je
    data = transform(p1, q1)
    axis_from = [ax.plot(data[i][0], data[i][1], data[i][2], colors[i])[0] for i in range(3)]

    #? izracunaj poziciju koja ce se menjati tokom izvrsavanja programa (ista kao i pocetna na pocetku)
    axis_current = [ax.plot(data[i][0], data[i][1], data[i][2], colors[i])[0] for i in range(3)]

    #? izracunaj krajnju poziciju i iscrtaj je
    data = transform(p2, q2)
    axis_to = [ax.plot(data[i][0], data[i][1], data[i][2], colors[i])[0] for i in range(3)]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=20, azim=-10)

    def animate(frame):
        #? transliraj ose
        p = lerp(p1, p2, tm, frame)

        #? nadji trenutni vektor rotacije
        q = slerp(q1, q2, tm, frame)

        data = transform(p, q)
        for i in range(len(axis_current)):
            
            #? postaviti novi polozaj objekta  
            axis_current[i].set_data(data[i][0], data[i][1])
            axis_current[i].set_3d_properties(data[i][2])

        fig.canvas.draw()

    anim = animation.FuncAnimation(fig, animate, frames=tm, interval=20, repeat=True, repeat_delay=200)

    save = True
    if save:
        anim.save('animation.gif', fps=20)

    plt.show()