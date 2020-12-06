import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import izometrije as ppgr

# quaternion product
def qmul(q1, q2):
    v1 = q1[0:3]
    w1 = q1[3]
    v2 = q2[0:3]
    w2 = q2[3]
    v = np.cross(v1, v2) + w2 * v1 + w1 * v2
    w = w1 * w2 - np.dot(v1, v2)
    return vec2q(v, w)

# quaternion conjugate
def qconj(q):
    return ppgr.make_quaternion(-q[0], -q[1], -q[2], q[3])

# quaternion inverse
def qinv(q):
    return qconj(q)/(ppgr.norm(q)**2)

# convert a 3d vector to a quaternion
def vec2q(v, w=0):
    if len(v) != 3:
        print('Error: Invalid vector dimension (not 3)')
        print(v)
        sys.exit()
    return ppgr.make_quaternion(v[0], v[1], v[2], w)

# for given position vector and roatation quaternion return plt data for the axes
def apply_transform(pos, q):
    qi = qinv(q)
    x = vec2q([1, 0, 0])
    x = qmul(qmul(q, x), qi)
    x = np.array([[pos[i], pos[i]+x[i]] for i in range(3)])

    y = vec2q([0, 1, 0])
    y = qmul(qmul(q, y), qi)
    y = np.array([[pos[i], pos[i]+y[i]] for i in range(3)])

    z = vec2q([0, 0, 1])
    z = qmul(qmul(q, z), qi)
    z = np.array([[pos[i], pos[i]+z[i]] for i in range(3)])
    return (x, y, z)

def lerp(q1, q2, tm, t):
    q = (1-(t/tm))*q1 + (t/tm)*q2
    return q

def slerp(q1, q2, tm, t):
    cos0 = np.dot(q1, q2)
    if cos0 < 0:
        q1 = -1 * q1
        cos0 = -cos0
    if cos0 > 0.95:
        return lerp(q1, q2, tm, t)
    angle = math.acos(cos0)
    q = (math.sin(angle*(1-t/tm)/math.sin(angle)))*q1 + (math.sin(angle*(t/tm)/math.sin(angle)))*q2
    return q

def euler2q(phi, theta, psi):
    phi_sin = math.sin(phi/2)
    phi_cos = math.cos(phi/2)
    theta_sin = math.sin(theta/2)
    theta_cos = math.cos(theta/2)
    psi_sin = math.sin(psi/2)
    psi_cos = math.cos(psi/2)
    return ppgr.make_quaternion(
        phi_sin*theta_cos*psi_cos - phi_cos*theta_sin*psi_sin,
        phi_cos*theta_sin*psi_cos + phi_sin*theta_cos*psi_sin,
        phi_cos*theta_cos*psi_sin - phi_sin*theta_sin*psi_cos,
        phi_cos*theta_cos*psi_cos + phi_sin*theta_sin*psi_sin
    )
    # Alternative
    # p, angle = ppgr.axis_angle(ppgr.euler2a(phi, theta, psi))
    # return axisangle2q(p, angle)

if __name__ == '__main__':
    # Input parameters:
    # Total number of frames
    tm = 100
    # Start position and rotation
    p1 = ppgr.make_vector([0.5, -1, 0])
    q1 = euler2q(math.radians(32), math.radians(-30), math.radians(75))
    # End position and rotation
    p2 = ppgr.make_vector([0.5, 0.9, 0.75])
    q2 = euler2q(math.radians(145), math.radians(-60), math.radians(-25))
    # Should save the animation
    save = False

    print('q1: ', q1)
    print('q2: ', q2)


    p = p1
    q = q1

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Disable the grid. For faster rendering
    # ax.axis('off')

    colors = ['r-', 'g-', 'b-']

    r = apply_transform(p1, q1)
    axis_from = [ax.plot(r[i][0], r[i][1], r[i][2], colors[i])[0] for i in range(3)]

    axis_current = [ax.plot(r[i][0], r[i][1], r[i][2], colors[i])[0] for i in range(3)]
    print('axis_current: ', axis_current)

    r = apply_transform(p2, q2)
    axis_to = [ax.plot(r[i][0], r[i][1], r[i][2], colors[i])[0] for i in range(3)]

    # Axis limits
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    ax.set_zlim((-4, 4))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set point of view
    ax.view_init(elev=20, azim=-10)

    def animate(frame):
        p = lerp(p1, p2, tm, frame)
        q = slerp(q1, q2, tm, frame)
        r = apply_transform(p, q)
        for i in range(len(axis_current)):

            axis_current[i].set_data(r[i][0], r[i][1])
            axis_current[i].set_3d_properties(r[i][2])

        fig.canvas.draw()

    #print('axis_current: ', axis_current)
    anim = animation.FuncAnimation(fig, animate, frames=tm, interval=20, repeat=True, repeat_delay=200)

    if save:
    #     # To save as a .gif the PIL/Pillow is required
            anim.save('animation.gif', fps=20)

    #     # It can also save as .mp4 if you have ffmpeg installed
    #     # anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()