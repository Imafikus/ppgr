import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image

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
    
    P_matrix = p2_matrix.dot(np.linalg.inv(p1_matrix))

    P_matrix = P_matrix / np.sum(P_matrix)
    print("######## NAIVE ########")
    print("Projective matrix for naive matrix, rounded on the 5th decimal")
    print(P_matrix.round(decimals=5))
    print()
    
    
    print("Check point D: ")
    D = np.array(points[0]) * alpha + np.array(points[1]) * beta + np.array(points[2]) * gamma
    print(D.round(decimals=5))

    return P_matrix

def remove_distortion_naive(points, projected_points, img_path_left, img_path_right):
    img_right = Image.open(img_path_right)
    img_left = Image.open(img_path_left)
    img_stiched = Image.new('RGB', (img_left.size[0], img_left.size[1]), "black")        
    # img_stiched = img_left.copy()
    
    P = get_projective_matrix_naive(points, projected_points)
    P_inverse = np.linalg.inv(P)
    cols = img_left.size[0]
    rows = img_left.size[1]

    for i in range(cols):        
        for j in range(rows):  
            
            new_coordinates = P_inverse.dot([i, j, 1])
            new_coordinates = [(x / new_coordinates[2]) for x in new_coordinates]
            
            if (new_coordinates[0] >= 0 and new_coordinates[0] < cols-1 and new_coordinates[1] >= 0 and new_coordinates[1] < rows-1):
                tmp2 = img_right.getpixel((math.ceil(new_coordinates[0]), math.ceil(new_coordinates[1])))
                img_stiched.putpixel((i, j), tmp2)

    for i in range(cols):        
        for j in range(rows):  
            if img_stiched.getpixel((i, j)) == (0, 0, 0):
                tmp2 = img_left.getpixel((i, j))
                img_stiched.putpixel((i, j), tmp2)
    
    
    fig = plt.figure(figsize = (16, 9))

    plt.subplot(2, 2, 1)
    plt.imshow(img_left)
    plt.title('Left Image')

    plt.subplot(2, 2, 2)
    plt.imshow(img_right)
    plt.title('Right Image')

    plt.subplot(2, 2, 3)
    plt.imshow(img_stiched)
    plt.title('Stiched Image')

    plt.tight_layout()
    plt.show()

def main():     

    #? Put different img path here if you want another image
    img_path_left = 'panorama1.jpg'
    img_path_right = 'panorama2.jpg'

    points = [
        [1449, 2192, 1],
        [1157, 2202, 1],
        [1543, 802, 1],
        [969, 1384, 1],
    ]

    projected_points = [
        [2521, 2174, 1],
        [2184, 2160, 1],
        [2679, 619, 1],
        [2000, 1291, 1],
    ]

    remove_distortion_naive(points, projected_points, img_path_left, img_path_right)

if __name__ == "__main__":
    main()