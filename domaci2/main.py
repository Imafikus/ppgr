import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image


chosen_points = []
chosen_projected_points = []

def register_mouse_click(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_points) >= 2:
            print("You've already selected 2 points, please restart the program if you want a new choice, or just close the images to continue")
            return

        print(f'Point chosen: {x}, {y}')
        chosen_points.append([x, y, 1])

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('Marked Image', img) 
        cv2.moveWindow('Marked Image', 600, 100)

def register_mouse_click_proj(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_projected_points) >= 4:
            print("You've already selected 4 points, please restart the program if you want a new choice, or just close the images to continue")
            return

        print(f'Point chosen: {x}, {y}')
        chosen_projected_points.append([x, y, 1])

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (0, 255, 0), 2) 
        cv2.imshow('Marked Image Proj', img) 
        cv2.moveWindow('Marked Image Proj', 600, 100)

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
    """
    Returns homogeneous coordinates for desired undistorted rectangle
    """
    rect_point_1 = points[0]
    rect_point_3 = points[1]
    
    rect_point_2 = [rect_point_1[0], rect_point_3[1], 1]
    rect_point_4 = [rect_point_3[0], rect_point_1[1], 1]

    chosen_points = [rect_point_1, rect_point_2, rect_point_3, rect_point_4]
    return sort_4_points(chosen_points)

def draw_desired_rectangle(img, points, projected_points):
    green = (0, 255, 0)
    red = ((0, 0, 255))
    thickness = 2

    cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), green, thickness=thickness)
    cv2.line(img, (points[1][0], points[1][1]), (points[2][0], points[2][1]), green, thickness=thickness)
    cv2.line(img, (points[2][0], points[2][1]), (points[3][0], points[3][1]), green, thickness=thickness)
    cv2.line(img, (points[3][0], points[3][1]), (points[0][0], points[0][1]), green, thickness=thickness)

    cv2.line(img, (projected_points[0][0], projected_points[0][1]), (projected_points[1][0], projected_points[1][1]), red, thickness=thickness)
    cv2.line(img, (projected_points[1][0], projected_points[1][1]), (projected_points[2][0], projected_points[2][1]), red, thickness=thickness)
    cv2.line(img, (projected_points[2][0], projected_points[2][1]), (projected_points[3][0], projected_points[3][1]), red, thickness=thickness)
    cv2.line(img, (projected_points[3][0], projected_points[3][1]), (projected_points[0][0], projected_points[0][1]), red, thickness=thickness)

    cv2.imshow('Desired rectangle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def remove_distortion_naive(points, projected_points, img_path):
    img = Image.open(img_path)
    img_copy = Image.new('RGB', (img.size[0], img.size[1]), "black")
    
    P_matrix = get_projective_matrix_naive(points, projected_points)
    P_matrix_inv = np.linalg.inv(P_matrix)
    
    cols = img_copy.size[0]
    rows = img_copy.size[1]

    for i in range(cols):
        for j in range(rows):
            new_coordinates = P_matrix_inv.dot([i, j, 1])
            new_coordinates = [(x / new_coordinates[2]) for x in new_coordinates]

            #? Check boundaries
            if (new_coordinates[0] >= 0 and new_coordinates[0] < cols-1 and new_coordinates[1] >= 0 and new_coordinates[1] < rows-1):
                new_pixel_value = img.getpixel((math.ceil(new_coordinates[0]), math.ceil(new_coordinates[1])))
                img_copy.putpixel((i, j), new_pixel_value)
    

    fig = plt.figure(figsize = (16, 9))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input')

    plt.subplot(1, 2, 2)
    plt.imshow(img_copy)
    plt.title('Output Naive')

    plt.tight_layout()
    plt.show()

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

def main():     

    # img_path = 'box.jpg'

    # img = cv2.imread(img_path)
    # img1 = img
    # img2 = img

    # print ("Please select 4 points in the image for distorted rectangle")

    # cv2.imshow('Choose distorted point', img)
    # cv2.setMouseCallback('Choose distorted point', register_mouse_click_proj, img1)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print ("Please select 2 diagonal points in the image for normal rectangle")

    # cv2.imshow('Choose normal point', img)
    # cv2.setMouseCallback('Choose normal point', register_mouse_click, img2)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sorted_chosen_projected_points = sort_4_points(chosen_projected_points)
    # sorted_chosen_points = sorted(chosen_points, key=lambda e: e[0])
    # sorted_chosen_points = determine_all_chosen_points(sorted_chosen_points)

    # print("Chosen distorted points: ", sorted_chosen_projected_points)
    # print("Chosen normal points: ", sorted_chosen_points)

    

    # points = [
    #     [358, 372, 1], 
    #     [372, 627, 1], 
    #     [766, 262, 1],
    #     [747, 493, 1] 
    # ]

    # projected_points = [
    #     [364, 371, 1], 
    #     [364, 495, 1], 
    #     [747, 371, 1], 
    #     [747, 495, 1]
    # ]

    # draw_naive_pic(sorted_chosen_points, sorted_chosen_projected_points)
    draw_naive_pic(None, None)


    # draw_naive(points, projected_points)
    # draw_desired_rectangle(img, sorted_chosen_points, sorted_chosen_projected_points)
    # remove_distortion_naive(sorted_chosen_points, sorted_chosen_projected_points, img, img_path)
    # remove_distortion_naive(projected_points, points, img_path)

def draw_naive_pic(points, points_proj):
    img = Image.open("box.jpg")
    img_copy = Image.new('RGB', (img.size[0], img.size[1]), "black")
    print("Image size: {} x {}".format(img.size[1], img.size[0]))
            
    # box.jpg
    points = [
        [358, 372, 1], 
        [372, 627, 1], 
        [766, 262, 1],
        [747, 493, 1] 
    ]

    points_proj = [
        [364, 371, 1], 
        [364, 495, 1], 
        [747, 371, 1], 
        [747, 495, 1]
    ]

    P = get_projective_matrix_naive(points, points_proj)

    P_inverse = np.linalg.inv(P)
    cols = img_copy.size[0]
    rows = img_copy.size[1]

    for i in range(cols):        
        for j in range(rows):  
            
            new_coordinates = P_inverse.dot([i, j, 1]) # lambda * X' = P * X
            new_coordinates = [(x / new_coordinates[2]) for x in new_coordinates]
            
            if (new_coordinates[0] >= 0 and new_coordinates[0] < cols-1 and            new_coordinates[1] >= 0 and new_coordinates[1] < rows-1):
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

if __name__ == "__main__":

    #Odabrati 4 tacke za projekciju, odabrati 2 tacke za pravougaonik, sortirati, izgenerisati 4 projektivne tacke, primeniti algoritam

    main()
    # print(sort_4_points([[361, 375, 1], [748, 495, 1], [762, 260, 1], [371, 625, 1]]))
    # draw_naive_pic()