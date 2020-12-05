import sys
import math
import numpy as np
import numpy.linalg as LA

# Helper function for calculating the norm of an any length vector
def norm(v):
    return math.sqrt(sum([x**2 for x in v]))

# Helper function for normalizing an any length vector
def normalize(v):
    return v/norm(v)

def make_vector(l):
    return np.array(l)

def make_quaternion(x, y, z, w):
    return np.array([x, y, z, w])

def create_rotation_matrix_x(angle):
    return np.array([
        [1,               0,                0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle),  math.cos(angle)]
    ])

def create_rotation_matrix_y(angle):
    return np.array([
        [ math.cos(angle), 0, math.sin(angle)],
        [               0, 1,               0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])

def create_rotation_matrix_z(angle):
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle),  math.cos(angle), 0],
        [              0,                0, 1]
    ])

def euler2a(phi, theta, psi):
    # Alternative
    # Rx = create_rotation_matrix_x(phi)
    # Ry = create_rotation_matrix_y(theta)
    # Rz = create_rotation_matrix_z(psi)
    # return np.matmul(np.matmul(Rz, Ry), Rx)
    return np.array([
        [
            math.cos(theta)*math.cos(psi),
            math.cos(psi)*math.sin(theta)*math.sin(phi) - math.cos(phi)*math.sin(psi),
            math.cos(phi)*math.cos(psi)*math.sin(theta) + math.sin(phi)*math.sin(psi)
        ],
        [
            math.cos(theta)*math.sin(psi),
            math.cos(phi)*math.cos(psi) + math.sin(theta)*math.sin(phi)*math.sin(psi),
            math.cos(phi)*math.sin(theta)*math.sin(psi) - math.cos(psi)*math.sin(phi)
        ],
        [
            -math.sin(theta),
            math.cos(theta)*math.sin(phi),
            math.cos(theta)*math.cos(phi)
        ]
    ])

def a2euler(A):
    # Because of floating point numbers we can't test for equality
    # instead we check for a small delta around 1
    if abs(LA.det(A)-1) >= 0.00001:
        print('Error: Determinant is not equal to 1 for matrix:')
        print(A)
        sys.exit()
    if (np.matmul(A.T, A) != np.eye(3)).all():
        print('Error: Matrix is not orthogonal:')
        print(A)
        sys.exit()

    if A[2, 0] < 1:
        if A[2, 0] > -1:
            psi = math.atan2(A[1,0], A[0,0])
            theta = math.asin(-A[2,0])
            phi = math.atan2(A[2,1], A[2,2])
        else:
            psi = math.atan2(-A[0,1], A[1,1])
            theta = math.pi/2
            phi = 0
    else:
        psi = math.atan2(-A[0,1], A[1,1])
        theta = -math.pi/2
        phi = 0
    
    return (phi, theta, psi)

def axis_angle(A):
    # Because of floating point numbers we can't test for equality
    # instead we check for a small delta around 1
    if abs(LA.det(A)-1) >= 0.00001:
        print('Error: Determinant is not equal to 1 for matrix:')
        print(A)
        sys.exit()
    if (np.matmul(A.T, A) != np.eye(3)).all():
        print('Error: Matrix is not orthogonal:')
        print(A)
        sys.exit()
    
    # (A - \E)p = 0
    # Solve for p when \ = 1
    B = A - np.eye(3)
    # Find 2 linearly independent vectors and return their cross product normalized
    p = np.cross(B[0], B[1])
    if not np.any(p):
        p = np.cross(B[0], B[2])
        if not np.any(p):
            p = np.cross(B[1], B[2])
    p = normalize(p)

    # x is any non-zero vector normal to p
    # so we choose one from matrix B and normalize it
    x = B[0]
    if not np.any(x):
        x = B[1]
        if not np.any(x):
            x = B[2]
    x = normalize(x)
    # xp is the result of moving x using A
    xp = np.matmul(A, x)

    # calculate the angle between x and xp
    angle = math.acos(np.dot(x, xp))
    if LA.det(np.array([x, xp, p])) < 0:
        p = -p
    
    return (p, angle)

def rodrigues(p, angle):
    p = normalize(p)
    ppt = p.reshape(3, -1) * p
    px = np.array([
        [0, -p[2], p[1]], 
        [p[2], 0, -p[0]], 
        [-p[1], p[0], 0]
    ])
    R = ppt + np.cos(angle)*(np.eye(3)-ppt) + np.sin(angle)*px
    return R

def axisangle2q(p, angle):
    w = math.cos(angle/2)
    p = normalize(p)
    x, y, z = math.sin(angle/2)*p
    return make_quaternion(x, y, z, w)

def q2axisangle(q):
    q = normalize(q)
    if q[3] < 0:
        q = -1 * q
    angle = 2 * math.acos(q[3])
    if abs(q[3]) == 1:
        p = make_vector([1, 0, 0])
    else:
        p = make_vector(q[0:3])
        p = normalize(p)
    return (p, angle)

if __name__ == '__main__':
    # phi, psi e [0, 2*PI]
    # theta e [-PI/2, PI/2]

    # Given test example
    # phi = -math.atan(1/4)
    # theta = -math.asin(8/9)
    # psi = math.atan(4)

    # My test example
    phi = 75
    theta = 60
    psi = 45
    print('degrees:')
    print(f'phi: {phi}, theta: {theta}, psi: {psi}')

    phi = math.radians(phi)
    theta = math.radians(theta)
    psi = math.radians(psi)
    print('radians:')
    print(f'phi: {phi}, theta: {theta}, psi: {psi}')

    print('euler2a:')
    A = euler2a(phi, theta, psi)
    print(A)

    print('axis angle:')
    p, angle = axis_angle(A)
    print(f'p: {p} angle: {angle}')

    print('rodrigues:')
    A = rodrigues(p, angle)
    print(A)

    print('a2euler:')
    phi, theta, psi = a2euler(A)
    print(f'phi: {phi}, theta: {theta}, psi: {psi}')

    print('axisangle2q:')
    q = axisangle2q(p, angle)
    print(q)

    print('q2axisangle:')
    p, angle = q2axisangle(q)
    print(f'p: {p} angle: {angle}')