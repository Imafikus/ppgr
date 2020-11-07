import numpy as np
import cv2

chosen_points = []

def register_mouse_click(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_points) >= 4:
            print("You've already selected 4 points, please restart the program if you want a new choice, or just close the images to continue")
            return

        print(f'Point chosen: {x}, {y}')
        chosen_points.append([x, y, 1])

        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('Marked Image', img) 
        cv2.moveWindow('Marked Image', 600, 100)

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


def determine_desired_coordinates(sorted_points): 
    """
    Returns homogeneous coordinates for desired undistorted desired
    """
    rect_point_1 = sorted_points[0]
    rect_point_2 = [sorted_points[0][0], sorted_points[1][1], 1]

    old_point_3 = sorted_points[3]

    rect_point_3 = [old_point_3[0], rect_point_1[1], 1]
    rect_point_4 = [rect_point_3[0], rect_point_2[1], 1]

    return [rect_point_1, rect_point_2, rect_point_3, rect_point_4]

def draw_desired_rectangle(img, points):
    color = (0, 255, 0)
    thickness = 2

    cv2.line(img, points[0], points[1], color, thickness=thickness)
    cv2.line(img, points[1], points[3], color, thickness=thickness)
    cv2.line(img, points[3], points[2], color, thickness=thickness)
    cv2.line(img, points[0], points[2], color, thickness=thickness)

    cv2.imshow('Desired rectangle', img)

def main():     
    img = cv2.imread('primer1.jpg')

    print ("Please select four points in the image")

    window_name = 'window'
    cv2.imshow(window_name, img)
    
    cv2.setMouseCallback(window_name, register_mouse_click, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Chosen points: ", chosen_points)
    sorted_chosen_points = sorted(chosen_points, key=lambda e: (e[0], e[1]))

    print("Sorted chosen_points: ", sorted_chosen_points)
    
    desired_coordinates = determine_desired_coordinates(sorted_chosen_points)
    print("Desired coordinates: ", desired_coordinates)

    get_projective_matrix_naive(sorted_chosen_points, desired_coordinates)
    
    # draw_desired_rectangle(img, rectangle_coordinates)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()