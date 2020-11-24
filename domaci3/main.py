import numpy as np

def is_identity_matrix(m):
    id = np.eye(3)
    m_id = m.dot(m.transpose())

    return (np.allclose(id, m_id))

def check_matrix(m):
    return is_identity_matrix(m) and int(np.linalg.det(m)) != 1

def get_normalized_vector(v):
    return v / np.sqrt(np.sum(v**2))


def Euler2A(phi, theta, psi):

    Rz = np.array([
        [ np.cos(psi), -np.sin(psi), 0],
        [ np.sin(psi), np.cos(psi), 0],
        [ 0, 0, 1],
    ])

    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0, 1, 0],
        [ -np.sin(theta), 0, np.cos(theta)],
    ])

    Rx = np.array([
        [ 1, 0, 0],
        [ 0, np.cos(phi), -np.sin(phi)],
        [ 0, np.sin(phi), np.cos(phi)],
    ])

    A = (Rz.dot(Ry)).dot(Rx)

    print('=== Euler2A ===')
    print('A: ', A)
    print()


    return A

def AxisAngle(A):
    if not is_identity_matrix(A) or np.linalg.det(A) != 1.0:
        print('Invalid matrix provided, A must be ortogonal and det(A) must be 1')
        return
    

def Rodrigez(p, phi):
    # print('p: ', p)
    # print()
    pT = np.matrix(p).T #? obican transpose ne radi lepo sa 1D matricama
    # print('pT: ', pT)
    # print()

    ppT = pT.dot(np.matrix(p)) #?P zadaje se kao niz, a treba nam matrica da bismo mogli da izmnozimo
    # print('ppT: ', ppT)
    # print()
    
    px = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0],
    ])
    # print('px: ', px)
    # print()

    Rp = ppT + np.cos(phi) * (np.eye(3) - ppT) + np.sin(phi) * px

    print('=== Rodrigez ===')
    print('Rp: ', Rp)
    print()

def A2Euler(A):
    if not check_matrix(A):
        print('Invalid matrix provided, A must be ortogonal and det(A) must be 1')
        return


    euler_angles = None
    if A[2][0] < 1:
        if A[2][0] > -1: 
            psi = np.arctan2(A[1][0], A[0][0])
            theta = np.arcsin(-A[2][0])
            phi = np.arctan2(A[2][1], A[2][2])
            euler_angles = [phi, theta, psi]
        else:
            psi = np.arctan2(A[1][0], A[1][1])
            theta = np.pi / 2
            phi = 0
            euler_angles = [phi, theta, psi]
    else:
        psi = np.arctan2(A[1][0], A[1][1])
        theta = np.pi / 2
        phi = 0
        euler_angles = [phi, theta, psi]

    print('=== A2Euler ===')
    print('euler_angles: ', euler_angles)
    print()

    return np.array(euler_angles)


def AxisAngle2Q(p, phi):
    normalized_p = get_normalized_vector(p)
    w = np.cos(phi / 2)

    im = np.sin(phi / 2) * normalized_p

    q = np.append(im, w)
    print('=== AxisAngle2Q ===')
    print('quaternion: ', q)
    print()

    return q

def Q2AngleAxis(q):
    q = get_normalized_vector(q)
    w = q[3]

    phi = 2 * np.arccos(w)

    if np.abs(w) == 1.0:
        p = np.array([1, 0, 0])
    else:
        p = get_normalized_vector(np.array(q[:-1]))

    print('=== Q2AngleAxis ===')
    print('p: ', p)
    print('phi: ', phi)
    print()

    return p, phi

def main():
    phi = -np.arctan(1/4)
    theta = -np.arcsin(8/9)
    psi = np.arctan(4)
    starting_angles = np.array([phi, theta, psi])

    A = Euler2A(phi, theta, psi)

    print('phi: ', phi)
    print('theta: ', theta)
    print('psi: ', psi)

    p = np.array([1 / 3, -2 / 3, 2 / 3])
    
    Rp = Rodrigez(p, np.pi / 2)
    # print(Rodrigez((np.sqrt(2)/2) * np.array([1, 1, 0]), np.pi / 3))

    a2_euler_angles = A2Euler(A)
    print('Compare starting_angles and euler_angles: ', starting_angles == a2_euler_angles)


    q = AxisAngle2Q(p, np.pi / 2)
    
    p_q2_angle_axis, phi = Q2AngleAxis(q)
    print('Compare starting p and p_q2_angle_axis: ', p == p_q2_angle_axis)


if __name__ == "__main__":
    main()