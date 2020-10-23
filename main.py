import numpy as np
import cv2

def display_image(img):
    cv2.imshow('Test img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_inf_point(a, b, c, d):
    inf_point = np.cross(np.cross(a, b), np.cross(c, d))
    inf_point = inf_point / inf_point[2]
    inf_point = inf_point.astype(int)
    
    return inf_point[0], inf_point[1]

def get_inf_point_homogen(a, b, c, d):
    inf_point = np.cross(np.cross(a, b), np.cross(c, d))
    return inf_point.tolist()

def main():     
    p1 = [757, 730, 1]
    p2 = [519, 594, 1]
    p3 = [713, 937, 1]
    p4 = [493, 1222, 1]
    p5 = [478, 1080, 1]
    p6 = [302, 1091, 1]
    p7 = [233, 926, 1]
    p8 = [509, 43, 1]

    img = cv2.imread('domaci1_original.jpg')

    xB = get_inf_point(p1, p3, p7, p6)    
    # draw lines related to xB
    img = cv2.line(img, (p1[0], p1[1]), xB, (255, 0, 0), 2)
    img = cv2.line(img, (p7[0], p7[1]), xB, (255, 0, 0), 2)
    img = cv2.line(img, (p2[0], p2[1]), xB, (0, 0, 255), 2)
    img = cv2.line(img, (p5[0], p5[1]), xB, (255, 0, 0), 2)

    yB = get_inf_point(p7, p2, p5, p1)
    # draw lines related to xB
    img = cv2.line(img, (p7[0], p7[1]), yB, (0, 255, 0), 2)
    img = cv2.line(img, (p5[0], p5[1]), yB, (0, 0, 255), 2)
    img = cv2.line(img, (p4[0], p4[1]), yB, (0, 255, 0), 2)
    img = cv2.line(img, (p6[0], p6[1]), yB, (0, 255, 0), 2)

    cv2.imwrite('domaci1_draw.jpg', img)

    xB_calc = get_inf_point_homogen(p1, p3, p7, p6)    
    print('xB_calc: ', xB_calc)
    yB_calc = get_inf_point_homogen(p2, p7, p1, p5)
    print('yB_calc: ', yB_calc)

    p_inv = get_inf_point(p2, xB_calc, p1, yB_calc)
    print(p_inv)


if __name__ == "__main__":
    main()