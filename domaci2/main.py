import numpy as np
import cv2

chosen_points = []

def register_mouse_click(event, x, y, flags, img):

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(chosen_points) >= 4:
            print("You've already selected 4 points, please restart the program if you want a new choice")
            return

        
        print(f'Point chosen: {x}, {y}')
        chosen_points.append((x, y))

        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('Marked Image', img) 
        cv2.moveWindow('Marked Image', 600, 100)

def main():     
    img = cv2.imread('primer1.jpg')

    print ("Please select four points in the image")

    window_name = 'window'
    cv2.imshow(window_name, img)
    
    cv2.setMouseCallback(window_name, register_mouse_click, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Chosen points: ", chosen_points)


if __name__ == "__main__":
    main()