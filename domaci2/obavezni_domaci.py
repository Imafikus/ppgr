import numpy as np
import math

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
    return (P.round(decimals=5) / P.sum(), alpha, beta, gamma)

def test_projective_matrix_naive(points, projected_points): #? Funkcija koja ispisuje matricu dobijenu naivnim algoritmom
    P_matrix, alpha, beta, gamma = get_projective_matrix_naive(points, projected_points)
    
    print("NAIVE")
    print("Projective matrix for naive algorithm, rounded on the 5th decimal")
    print(P_matrix)
    print()
    
    
    print("Check point D: ") #? Proveravamo da li smo zaista dobro resili sistem
    D = np.array(points[0]) * alpha + np.array(points[1]) * beta + np.array(points[2]) * gamma
    print(D.round(decimals=5))

def get_projective_matrix_dlt(points, projected_points, get_normalized = True):#? Uvek zelim da default ponasanje bude normalizacija matrice kako bismo mogli lako da uporedimo razlicite algoritme
    main_matrix = []
    n = len(points)
    for i in range(n):
        main_matrix.append([
            0, 0, 0, 
            -projected_points[i][2]*points[i][0], -projected_points[i][2]*points[i][1], -projected_points[i][2]*points[i][2], 
            projected_points[i][1]*points[i][0], projected_points[i][1]*points[i][1], projected_points[i][1]*points[i][2]
        ])     

        main_matrix.append([
            projected_points[i][2]*points[i][0], projected_points[i][2]*points[i][1], projected_points[i][2]*points[i][2],
            0, 0, 0,
            -projected_points[i][0]*points[i][0], -projected_points[i][0]*points[i][1], -projected_points[i][0]*points[i][2]
        ])

    _, _, V = np.linalg.svd(main_matrix, full_matrices = True)
    P_matrix_DLT = V[8]*(-1)
    
    if get_normalized:
        return P_matrix_DLT.round(decimals=5).reshape((3, 3)) / P_matrix_DLT.sum()
    else:
        return P_matrix_DLT.reshape((3, 3))
        

def test_projective_matrix_dlt(points, projected_points): #? Ova funkcija samo racuna projektivno matricu naivnim algoritmom

    P_matrix = get_projective_matrix_dlt(points, projected_points)
    print()
    print("DLT:")
    print("Projective matrix for dlt algorithm, rounded on the 5th decimal")
    print(P_matrix)
    print()
    
def compare_dlt_naive(dlt_p, dlt_pp, naive_p, naive_pp): #? Ova funkcija racuna projektivnu matricu i za naivni i za obican DLT algoritam
    print("COMPARE DLT AND NAIVE")
    print('DLT is calculated on 6 points, Naive is calculated on 4 points, both are rounded and compared')
    P_matrix_DLT = get_projective_matrix_dlt(dlt_p, dlt_pp)
    P_matrix_naive, _, _, _ = get_projective_matrix_naive(naive_p, naive_pp)
    print(P_matrix_naive)
    print()

    print('Comparison of individual elements: ')
    #? Poredimo elemente i ako je dovoljno mala razlika (odnosno, racun je dobar), dobicemo matricu koja ce svuda biti True, 
    print(P_matrix_DLT.round() == P_matrix_naive.round())
    print()

def test_dlt_coordinates_scaled_vs_untouched_coordinates(points, projected_points): #? Testiramo sta se desi kada promenimo koordinate
    print("RESCALE COORDINATES FOR DLT")
    scaled_projected_points = normalize_points(projected_points)
    scaled_points = normalize_points(points)
    

    P_matrix = get_projective_matrix_dlt(points, projected_points, get_normalized=False)
    print("Projective matrix before rescale:")
    print(P_matrix)
    print()


    P_matrix_after_rescale = get_projective_matrix_dlt(scaled_points, scaled_projected_points, get_normalized=False)
    print("Projective matrix after rescale:")
    print(P_matrix_after_rescale)
    print()

def get_homo_coef(points):
    coef = sum([math.sqrt(p[0]*p[0] + p[1]*p[1]) for p in points]) / len(points)
    return coef

def get_normalized_points(points): #? Funkcija koja radi translaciju i skaliranje
    center_x = sum([p[0] for p in points]) / len(points)
    center_y = sum([p[1] for p in points]) / len(points)

    translated_points = [[p[0] - center_x, p[1] - center_y] for p in points]

    coef = get_homo_coef(translated_points)

    T_matrix = [
        [math.sqrt(2) / coef, 0, center_x * (-1)],
        [0, math.sqrt(2) / coef, center_y * (-1)],
        [0, 0, 1],
    ]

    return np.array(T_matrix)

def normalize_points(points):
    return [[x / z, y / z, 1] for [x, y, z] in points]

def get_projective_matrix_dlt_normalized(points, projected_points): #? Funkcija koja racuna projektivnu matricu prateci normalizovan DLT algoritam

    points = normalize_points(points)
    projected_points = normalize_points(projected_points)

    T_matrix = get_normalized_points(points)
    T_projected_matrix = get_normalized_points(projected_points)

    points = np.transpose(points)
    projected_points = np.transpose(projected_points)

    M = T_matrix.dot(points)
    M_proj = T_projected_matrix.dot(projected_points)

    M = np.transpose(M)
    M_proj = np.transpose(M_proj)

    dlt = get_projective_matrix_dlt(M, M_proj, get_normalized=False)

    res = (np.linalg.inv(T_projected_matrix)).dot(dlt).dot(T_matrix)

    return res, T_matrix, T_projected_matrix


def test_projective_matrix_dlt_normalized(points, projected_points): #? Funkcija koja ispistuje konacno matricu i matrice T i T'
    print("DLT NORMALIZED:")
    print()
    
    res, T_matrix, T_projected_matrix = get_projective_matrix_dlt_normalized(points, projected_points)
    
    print("DLT matrix calculated with 6 points, rounded on 5 decimals: ")
    print(res.round(decimals=5))
    print()

    print("T matrix calculated with 6 points: ")
    print(T_matrix)
    print()

    print("T' matrix calculated with 6 points: ")
    print(T_projected_matrix)
    print()

def rescale_matrix(matrix):
    return matrix / matrix[0][0]

def testing_algorithms():
    y1 = [0, 1, 1]
    y2 = [1, 1, 1]
    y3 = [-3, -1, 1]
    y4 = [-1, 2, 1]
    y5 = [2, 0, 1]

    y1p = [-1, 1, 1]
    y2p = [5, 0, 1]
    y3p = [1, -2, 1]
    y4p = [2, 3, 1]
    y5p = [4, 3, 1]

    yn1 = [-2, -1, 1]
    yn2 = [-1, -2, 1]
    yn3 = [-7, 0, 1]
    yn4 = [-2, 1, 1]
    yn5 = [-1, -4, 1]

    yn1p = [3, -2, 1]
    yn2p = [4, 4, 1]
    yn3p = [6, 0, 1]
    yn4p = [1, 1, 1]
    yn5p = [1, 3, 1]

    print("Naivni i DLT za y i yp: PRIMER 1")
    naive_matrix, _, _, _ = get_projective_matrix_naive([y1, y2, y3, y4], [y1p, y2p, y3p, y4p])
    naive_matrix = rescale_matrix(naive_matrix)
    dlt_matrix = get_projective_matrix_dlt([y1, y2, y3, y4], [y1p, y2p, y3p, y4p])
    dlt_matrix = rescale_matrix(dlt_matrix)

    print('NAIVE MATRIX: ')
    print(naive_matrix)
    print('DLT MATRIX')
    print(dlt_matrix)
    print()


    print("DLT i modifikovani DLT za y i yp: PRIMER 2")
    dlt_matrix = get_projective_matrix_dlt([y1, y2, y3, y4, y5], [y1p, y2p, y3p, y4p, y5p])
    dlt_matrix = rescale_matrix(dlt_matrix)

    dlt_matrix_normalized, _, _ = get_projective_matrix_dlt_normalized([y1, y2, y3, y4, y5], [y1p, y2p, y3p, y4p, y5p])
    dlt_matrix_normalized = rescale_matrix(dlt_matrix_normalized)

    print('DLT MATRIX: ')
    print(dlt_matrix)
    print('DLT MATRIX NORMALIZED')
    print(dlt_matrix_normalized)
    print()

    
    print("modifikovani DLT za yn i ynp: PRIMER 3")
    dlt_matrix_normalized, _, _ = get_projective_matrix_dlt_normalized([yn1, yn2, yn3, yn4, yn5], [yn1p, yn2p, yn3p, yn4p, yn5p])
    dlt_matrix_normalized = rescale_matrix(dlt_matrix_normalized)

    print('DLT MATRIX NORMALIZED')
    print(dlt_matrix_normalized)
    print()


def run_tests_and_comparisons():

    points_for_naive = [
        [-3, -1, 1],
        [3, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ]

    projected_points_for_naive = [
        [-2, -1, 1],
        [2, -1, 1],
        [2, 1, 1],
        [-2, 1, 1]
    ]

    points_for_dlt = [
        [-3, -1, 1],
        [3, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [1, 2, 3],
        [-8, -2 ,1]
    ]

    projected_points_for_dlt = [
        [-2, -1, 1],
        [2, -1, 1],
        [2, 1, 1],
        [-2, 1, 1],
        [2, 1, 4],
        [-16, -5, 4]
    ]

    test_projective_matrix_naive(points_for_naive, projected_points_for_naive)
    test_projective_matrix_dlt(points_for_dlt, projected_points_for_dlt)
    compare_dlt_naive(points_for_dlt, projected_points_for_dlt, points_for_naive, projected_points_for_naive)
    test_dlt_coordinates_scaled_vs_untouched_coordinates(points_for_dlt, projected_points_for_dlt)
    test_projective_matrix_dlt_normalized(points_for_dlt, projected_points_for_dlt)

if __name__ == "__main__":
    print("--------TESTING ALGORITHMS--------")
    testing_algorithms()
    print()

    print("--------TESTS AND COMPARISONS--------")
    run_tests_and_comparisons()
    print()
    
    