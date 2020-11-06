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

def nevidljivo(p1, p2, p3, p5, p6, p7, p8):
    p1.append(1)
    p2.append(1)
    p3.append(1)
    p5.append(1)
    p6.append(1)
    p7.append(1)
    p8.append(1)

    img = cv2.imread('domaci1_vol2_marked.jpeg')
    cv2.imshow('prozor', img)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    xB = get_inf_point(p2, p6, p1, p5)    
    # draw lines related to xB
    img = cv2.line(img, (p6[0], p6[1]), xB, (255, 0, 0), 2)
    img = cv2.line(img, (p7[0], p7[1]), xB, (255, 0, 0), 2)
    img = cv2.line(img, (p8[0], p8[1]), xB, (0, 0, 255), 2)
    img = cv2.line(img, (p5[0], p5[1]), xB, (255, 0, 0), 2)

    yB = get_inf_point(p5, p6, p7, p8)    
    # draw lines related to xB
    img = cv2.line(img, (p7[0], p7[1]), yB, (0, 255, 0), 2)
    img = cv2.line(img, (p2[0], p2[1]), yB, (0, 255, 0), 2)
    img = cv2.line(img, (p3[0], p3[1]), yB, (0, 0, 255), 2)
    img = cv2.line(img, (p6[0], p6[1]), yB, (0, 255, 0), 2)

    cv2.imwrite('domaci_vol2_draw.jpeg', img)
    
    xB_calc = get_inf_point_homogen(p2, p6, p1, p5)    
    yB_calc = get_inf_point_homogen(p5, p6, p7, p8)

    p4 = get_inf_point(p8, xB_calc, p3, yB_calc)
    print("Koordinate nevidljive tacke: ", p4)

def main():     
    p1 = [542, 663]
    p2 = [391, 771]
    p3 = [167, 693]
    p5 = [577, 277]
    p6 = [426, 309]
    p7 = [180, 270]
    p8 = [381, 248]

    nevidljivo(p1, p2, p3, p5, p6, p7, p8)

if __name__ == "__main__":
    main()