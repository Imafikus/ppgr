import numpy as np 

def get_camera_eq(m, mp):
    x1 = np.hstack((np.array([0,0,0,0]), -mp[2]*m, mp[1]*m))
    x2 = np.hstack((mp[2]*m, np.array([0,0,0,0]), -mp[0]*m))
    return np.array([x1, x2])

def cameraDLT(M, Mp):
    A = []

    for i in range(len(M)):
        eq = get_camera_eq(M[i], Mp[i])
        if i == 0:
            A = eq 
        else:
            A = np.vstack((A, eq))
    
    u, s, vh = np.linalg.svd(A)
    return vh[11].reshape(3, 4)


def main():
    n = 1 # 121 / 2017
    M1 = np.array([460, 280, 250, 1])
    M1p = np.array([288, 251, 1])

    M2 = np.array([50, 380, 350, 1])
    M2p = np.array([79, 510, 1])

    M3 = np.array([470, 500, 100, 1])
    M3p = np.array([470, 440, 1])

    M4 = np.array([380, 630, 50*n, 1])
    M4p = np.array([520, 590, 1])

    M5 = np.array([30*n, 290, 0, 1])
    M5p = np.array([365, 388, 1])

    M6 = np.array([580, 0, 130, 1])
    M6p = np.array([365, 20, 1])

    M = np.array([
        M1,
        M2,
        M3,
        M4,
        M5,
        M6,
    ])

    Mp = np.array([
        M1p,
        M2p,
        M3p,
        M4p,
        M5p,
        M6p,
    ])

    T = cameraDLT(M, Mp)
    T = T / T[0][0]
    
    np.set_printoptions(suppress=True)
    print('=== T ===\n', T)
    print()


if __name__ == "__main__":
    main()