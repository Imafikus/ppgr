import numpy as np

def is_identity_matrix(m):
    id = np.eye(3)
    m_id = m.dot(m.transpose())

    return (np.allclose(id, m_id))

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

    ppT = pT.dot(np.matrix(p)) #? Zadaje se kao niz, a treba nam matrica da bismo mogli da izmnozimo
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



def main():
    A = Euler2A(-np.arctan(1/4), -np.arcsin(8/9), np.arctan(4))
    
    Rp = Rodrigez(np.array([1 / 3, -2 / 3, 2 / 3]), np.pi / 2)
    # print(Rodrigez((np.sqrt(2)/2) * np.array([1, 1, 0]), np.pi / 3))

if __name__ == "__main__":
    main()