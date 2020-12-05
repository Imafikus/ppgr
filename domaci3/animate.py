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
    c_start = np.array([0.75, -1, 0])
    c_end = np.array([0.5, 0.8, 0.75])
    
    # Pocetna duzina stranica
    startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # pocetne rotacije
    q0 = np.array([0.43882373, 0.03532013, 0.62183644, 0.64769257])
    q1 = np.array([0.71725074, 0.17018643, -0.49442859,  0.46057391])

    # inicijalizacija animacije i iscrtavanja
    fig = plt.figure(figsize=(7,7))
    
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.set_zlim((-1.5, 1.5))
    
    ax.view_init(elev=20, azim=-10)
    

    colors = ['r', 'g', 'b']

    # iscrtavanje pocetne i kranje pozicije

    axis_current = []

    for i in range(3):
        # iscrtaj pocetnu poziciju
        start = transform(startpoints[i], q0).flatten() + c_start
        start = start.tolist()[0]

        end = transform(endpoints[i], q0) + c_start 
        end = end.tolist()[0]
        ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])

        axis_current.append(ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i]))

        #iscrtaj krajnju poziciju
        start = transform(startpoints[i], q1) + c_end
        start = start.tolist()[0]

        end = transform(endpoints[i], q1) + c_end
        end = end.tolist()[0]

        ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])
        
    lines = np.array(sum([ax.plot([], [], [], c=c) for c in colors], []))
    
    axis_current = np.array(axis_current).flatten().tolist()
    print(axis_current)

    # steps = [slerp(q0, q1, frames, i/frames) for i in range(frames)]

    # def init():
    #     for line in lines:
    #         line.set_data([], [])
    #         line.set_3d_properties([])
    #     return lines

    def animate(frame):
        q = slerp(q0, q1, frames, frame)
        print('q:  ', q)
        
        t = frame * (c_end - c_start) / frames + c_start
        print('t:  ', t)

        print('axis_current: ', axis_current)
        print('startpoints: ', startpoints)
        print('endpoints: ', endpoints)
        print('frames: ', frames)
        print('frame: ', frame)

        #exit()
        i = 0
        for i in range(len(axis_current)):
        
            start = transform(startpoints[i], q) + t
            start = start.tolist()[0]

            end = transform(endpoints[i], q) + t
            end = end.tolist()[0]   
            
            # ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])
            kurcina = np.array([start[2], end[2]])
            print('kurcina : ', kurcina)
            print('kurcina.shape : ', kurcina.shape)

            axis_current[i].set_data([start[0], end[0]], [start[1], end[1]])
            axis_current[i].set_3d_properties(start[2], end[2])
            i += 1
    #        fig.clear()
    
        return lines
        # fig.canvas.draw()


    anim = animation.FuncAnimation(fig, animate, frames=frames+1, interval=20, repeat=True, repeat_delay=200)

    plt.show()
