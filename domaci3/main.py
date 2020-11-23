import numpy as np

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

    return (Rz.dot(Ry)).dot(Rx)

def main():
    A = Euler2A(-np.arctan(1/4), -np.arcsin(8/9), np.arctan(4))
    print(A)

if __name__ == "__main__":
    main()