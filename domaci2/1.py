import numpy as np
import cv2

def solve_system(points):

    A, B, C, D = points

    d = np.float32([
        [A[0], B[0], C[0]],
        [A[1], B[1], C[1]],
        [A[2], B[2], C[2]]
    ])

    d1 = np.float32([
        [D[0], B[0], C[0]],
        [D[1], B[1], C[1]],
        [D[2], B[2], C[2]]
    ])

    d2 = np.float32([
        [A[0], D[0], C[0]],
        [A[1], D[1], C[1]],
        [A[2], D[2], C[2]]
    ])

    d3 = np.float32([
        [A[0], B[0], D[0]],
        [A[1], B[1], D[1]],
        [A[2], B[2], D[2]]
    ])
    
    d_det = (np.linalg.det(d))
    d1_det = (np.linalg.det(d1))
    d2_det = (np.linalg.det(d2))
    d3_det = (np.linalg.det(d3))

    
    alpha = d1_det / d_det
    beta = d2_det / d_det
    gamma = d3_det / d_det

    return (alpha, beta, gamma)

def get_projective_matrix_naive(points, projected_points):
    (alpha, beta, gamma) = solve_system(points)
    p1_matrix = np.array([
        [x*alpha for x in points[0]],
        [x*beta for x in points[1]],
        [x*gamma for x in points[2]]
    ])
    p1_matrix = np.transpose(p1_matrix)
    
    (alpha_p, beta_p, gamma_p) = solve_system(projected_points)
    p2_matrix = np.array([
        [x*alpha_p for x in projected_points[0]],
        [x*beta_p for x in projected_points[1]],
        [x*gamma_p for x in projected_points[2]]
    ])
    p2_matrix = np.transpose(p2_matrix)
    
    P = p2_matrix.dot(np.linalg.inv(p1_matrix))
    return (P, alpha, beta, gamma)

def test_projective_matrix_naive():
    points = [
        [1, 1, 1],
        [5, 2, 1],
        [6, 4, 1],
        [-1, 7, 1]
    ]

    projected_points = [
        [0, 0, 1],
        [10, 0, 1],
        [10, 5, 1],
        [0, 5, 1]
    ]

    P_matrix, alpha, beta, gamma = get_projective_matrix_naive(points, projected_points)
    P_matrix = P_matrix / np.sum(P_matrix)
    print("######## NAIVE ########")
    print("Projective matrix for naive matrix, rounded on the 5th decimal")
    print(P_matrix.round(decimals=5))
    print()
    
    
    print("Check point D: ")
    D = np.array(points[0]) * alpha + np.array(points[1]) * beta + np.array(points[2]) * gamma
    print(D.round(decimals=5))


if __name__ == "__main__":
    test_projective_matrix_naive()
