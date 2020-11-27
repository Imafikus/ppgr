import numpy as np
from scipy.sparse import linalg

def get_normalized_vector(v):
    return v / np.linalg.norm(v)

def check_matrix(A):
    id = np.eye(3).astype(int)

    AT = A.T

    if not (AT.dot(A).round() == id).all():
        return False 
    
    if np.linalg.det(A).round() != 1.0:
        return False 
    
    return True


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

    A = Rz.dot(Ry).dot(Rx)

    print('=== Euler2A ===')
    print('A: \n', A)
    print()

    return A

def AxisAngle(A):

    if not check_matrix(A):
        print('Invalid matrix provided')
        return
    
    _, p = linalg.eigs(A, k=1, sigma=1)
    p = p.flatten().astype(float)

    v = np.array([0, 0, 1])
    
    u = get_normalized_vector(np.cross(p, v))
    u_p = np.matmul(A, u)
    
    phi = np.arccos(np.dot(u, u_p))

    mixed_product = np.dot(u, np.cross(u_p, p))
    if mixed_product < 0:
        p = -p
    
    print()
    print('=== AxisAngle ===')
    print('p: ', p)
    print('phi: ', phi)
    print()

    return p, phi


def Rodrigez(p, phi):
    pT = np.matrix(p).T #? obican transpose ne radi lepo sa 1D matricama

    ppT = pT.dot(np.matrix(p)) #?p zadaje se kao niz, a treba nam matrica da bismo mogli da izmnozimo
    
    px = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0],
    ])

    Rp = ppT + np.cos(phi) * (np.eye(3) - ppT) + np.sin(phi) * px

    print('=== Rodrigez ===')
    print('Rp: \n', Rp)
    print()
    
    return Rp

def A2Euler(A):
    if not check_matrix(A):
        print('Invalid matrix provided')
        return

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

    norm_q = get_normalized_vector(q)
    
    w = q[3]

    if w < 0:
        q = -q
    
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

def show_functions(phi, theta, psi):
    starting_angles = np.array([phi, theta, psi])

    print('Starting angles: ')
    print('phi: ', phi)
    print('theta: ', theta)
    print('psi: ', psi)
    print()
    
    A = Euler2A(phi, theta, psi)

    p, new_phi = AxisAngle(A)    
    
    Rp = Rodrigez(p, new_phi)

    print('Compare Rp and A: \n', Rp == A)
    print()

    a2_euler_angles = A2Euler(A)
    print('Compare starting_angles and euler_angles: ', starting_angles == a2_euler_angles)
    print()

    q = AxisAngle2Q(p, new_phi)
    
    p_q2_angle_axis, phi = Q2AngleAxis(q)
    print('Compare starting p and p_q2_angle_axis: ', p.round() == p_q2_angle_axis.round())
    print()

def main():
    phi_test_case = -np.arctan(1/4)
    theta_test_case = -np.arcsin(8/9)
    psi_test_case = np.arctan(4)
    print('****** TEST CASE ******')
    print()
    show_functions(phi_test_case, theta_test_case, psi_test_case)
    print('****** END TEST CASE ******')
    print()

    phi_custom_case = np.pi / 3
    theta_custom_case = np.pi / 3
    psi_custom_case = np.pi / 3
    print('****** CUSTOM CASE ******')
    print()
    show_functions(phi_custom_case, theta_custom_case, psi_custom_case)
    print('****** END CUSTOM CASE ******')
    print()


if __name__ == "__main__":
    main()