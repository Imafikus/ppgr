import numpy as np

def correct_if_needed(Q, R):
    
    for d in range(0, 3):
        if R[d][d] < 0:
            R[d] = - R[d]
            for i in range(3):
                Q[i][d] = -Q[i][d]
    return Q, R

def calc_C(T):
    c1 = 1 * np.linalg.det(np.array([
        [T[0][1], T[0][2], T[0][3]],
        [T[1][1], T[1][2], T[1][3]],
        [T[2][1], T[2][2], T[2][3]],
    ]))

    c2 = -1 * np.linalg.det(np.array([
        [T[0][0], T[0][2], T[0][3]],
        [T[1][0], T[1][2], T[1][3]],
        [T[2][0], T[2][2], T[2][3]],
    ]))

    c3 = 1 * np.linalg.det(np.array([
        [T[0][0], T[0][1], T[0][3]],
        [T[1][0], T[1][1], T[1][3]],
        [T[2][0], T[2][1], T[2][3]],
    ]))

    c4 = -1 * np.linalg.det(np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]],
    ]))

    return np.array([[c1, c2, c3]]) / c4

def main():
    n = 1 # 121 / 2017
    T = np.array([
        [5, -1-2*n, 3, 18-3*n],
        [0,   -1,   5,   21],
        [0,   -1,   0,   1],
        ])

    T0 = np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]]
    ])

    if np.linalg.det(T0) < 0:
        T0 = -T0

    T0_inv = np.linalg.inv(T0)

    Q, R = np.linalg.qr(T0_inv)
    Q, R = correct_if_needed(Q, R)


    K = np.linalg.inv(R)
    K = K / K[2][2]

    A = np.linalg.inv(Q)

    C = calc_C(T)

    print('=== K ===\n', K)
    print()
    
    print('=== A ===\n', A)
    print()

        
    print('=== C ===\n', C)
    print()

if __name__ == "__main__":
    main()
