import numpy as np


def correct_if_needed(Q, R):
    
    for d in range(0, 3):
        if R[d][d] < 0:
            R[d] = - R[d]
            for i in range(3):
                Q[i][d] = -Q[i][d]
    return Q, R

def main():
    n = 11 # 121 / 2017
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
    print(T0_inv)

    Q, R = np.linalg.qr(T0_inv)
    Q, R = correct_if_needed(Q, R)


    K = np.linalg.inv(R)
    K = K / K[2][2]

    A = np.linalg.inv(Q)

    print('=== K ===\n', K)
    print()
    
    print('=== A ===\n', A)
    print()

if __name__ == "__main__":
    main()
