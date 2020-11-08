import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image


chosen_points = []
chosen_projected_points = []

def register_mouse_click(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_projected_points) >= 2:
            print("You've already selected 2 points, please restart the program if you want a new choice, or just close the images to continue")
            return

        print(f'Point chosen: {x}, {y}')
        chosen_projected_points.append([x, y, 1])

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('Marked image', img) 
        cv2.moveWindow('Marked image', 600, 100)

def register_mouse_click_proj(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_points) >= 4:
            print("You've already selected 4 points, please restart the program if you want a new choice, or just close the images to continue")
            return

        print(f'Point chosen: {x}, {y}')
        chosen_points.append([x, y, 1])

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (0, 255, 0), 2) 
        cv2.imshow('Marked image', img) 
        cv2.moveWindow('Marked image', 600, 100)

def sort_4_points(points):
    points = sorted(points, key=lambda e: (e[0]))
    p1, p2, p3, p4 = points

    if p1[1] > p2[1]:
        tmp = p2
        p2 = p1
        p1 = tmp
    
    if p3[1] > p4[1]:
        tmp = p4
        p4 = p3
        p3 = tmp

    return [p1, p2, p3, p4]

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

def determine_all_chosen_points(points): 
    rect_point_1 = points[0]
    rect_point_3 = points[1]
    
    rect_point_2 = [rect_point_1[0], rect_point_3[1], 1]
    rect_point_4 = [rect_point_3[0], rect_point_1[1], 1]

    chosen_points = [rect_point_1, rect_point_2, rect_point_3, rect_point_4]
    return sort_4_points(chosen_points)

def remove_distortion_naive(points, projected_points, img_path):
    img = Image.open(img_path)
    img_copy = Image.new('RGB', (img.size[0], img.size[1]), "black")
            
    #? Put custom points and projected_points here if you dont want to input them by clicking 
    #? After that just call this function with remove_distortion_naive(None, None, img_path) instead of calling main()

    P = get_projective_matrix_naive(points, projected_points)

    P_inverse = np.linalg.inv(P)
    cols = img_copy.size[0]
    rows = img_copy.size[1]

    for i in range(cols):        
        for j in range(rows):  
            
            new_coordinates = P_inverse.dot([i, j, 1])
            new_coordinates = [(x / new_coordinates[2]) for x in new_coordinates]
            
            if (new_coordinates[0] >= 0 and new_coordinates[0] < cols-1 and new_coordinates[1] >= 0 and new_coordinates[1] < rows-1):
                tmp1 = img.getpixel((math.floor(new_coordinates[0]), math.floor(new_coordinates[1])))
                tmp2 = img.getpixel((math.ceil(new_coordinates[0]), math.ceil(new_coordinates[1])))
                img_copy.putpixel((i, j), tmp2)
    fig = plt.figure(figsize = (16, 9))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input')

    plt.subplot(1, 2, 2)
    plt.imshow(img_copy)
    plt.title('Output Naive')

    plt.tight_layout()
    plt.show()

def main():     

    #? Put different img path here if you want another image
    img_path = 'spomenik.jpg'

    img = cv2.imread(img_path)
    img1 = img
    img2 = img

    print ("Please select 4 points in the image for distorted rectangle")

    cv2.imshow('Choose distorted point', img)
    cv2.setMouseCallback('Choose distorted point', register_mouse_click_proj, img1)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print ("Please select 2 diagonal points in the image for the projected rectangle, first point must in the upper left, and second must be in the lower right corner")

    cv2.imshow('Choose normal point', img)
    cv2.setMouseCallback('Choose normal point', register_mouse_click, img2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points = sort_4_points(chosen_points)
    projected_points = sorted(chosen_projected_points, key=lambda e: e[0])
    projected_points = determine_all_chosen_points(projected_points)

    print("Chosen distorted points: ", points)
    print("Chosen projected points: ", projected_points)

    remove_distortion_naive(points, projected_points, img_path)

if __name__ == "__main__":
    main()