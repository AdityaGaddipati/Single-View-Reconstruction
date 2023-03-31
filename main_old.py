from calendar import c
import enum
from glob import glob
import numpy as np
import cv2
import argparse

from q1 import q1_affine_rectify
from q2 import q2_metric_rectify
from q3 import compute_homography
from q4 import q4_metric_rectify
from utils import MyWarp, CalcAngle


annotations = []
def click_event(event, x, y, flags, params):
    global annotations
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(annotations)==0 or len(annotations[-1])==4:
            annotations.append([x,y])
        else:
            annotations[-1].append(x)
            annotations[-1].append(y)
            
        # print(annotations)    
            
        for l in annotations:
            if len(l) != 4:
                continue
            cv2.line(img_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), thickness=10)
        
        cv2.imshow('image', img_copy)


def display_img(img):
    global annotations
    annotations=[]
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
        
def draw_annotations(img, annotations):
    viz_img = img.copy()
    color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    for i,l in enumerate(annotations):
        cv2.line(viz_img, (l[0], l[1]), (l[2], l[3]), color[int(i/2)], thickness=10)
    # display_img(viz_img)

    return viz_img

def save_img(img, path):
    cv2.imwrite(path, img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=int, help = "question number")
    args = parser.parse_args()
    
    # input_img = cv2.imread('./data/q1/tiles5.JPG', 1) 
    # input_img = cv2.imread('./data/q1/chess1.jpg', 1) 
    # input_img = cv2.imread('./data/q1/checker1.jpg', 1) 
    # input_img = cv2.imread('./data/q1/facade.jpg', 1)
    # input_img = cv2.imread('./data/q1/book2.jpg', 1) 
    # input_img = cv2.imread('./data/q1/book1.jpg', 1)

    # input_img = cv2.imread('./data/my_images/church.jpg', 1)
    # input_img = cv2.imread('./data/my_images/cathedral.jpg', 1) 
    # input_img = cv2.imread('./data/my_images/tcs.jpg', 1) 
    input_img = cv2.imread('./data/my_images/garden.jpg', 1)


    # input_img = cv2.imread('./data/q3/desk-perspective.jpg', 1)
    # img1 = cv2.imread('./data/q3/desk-normal.jpg', 1)
    # input_img = cv2.imread('./data/q3/downtown.jpg', 1)
    # img1 = cv2.imread('./data/q3/sky.jpeg', 1)

    img_name = 'garden.jpg'

    img_copy = input_img.copy()
    display_img(img_copy)

    if args.question==1:
        H = q1_affine_rectify(annotations)
        
        annotated_img = draw_annotations(input_img,annotations)
        path = "output/q1/annotated_img/" + img_name
        save_img(annotated_img,path)

        result = MyWarp(input_img, H)
        path = "output/q1/rectified_image/" + img_name
        save_img(result,path)

        # Test line angles
        display_img(img_copy)
        test_lines_img = draw_annotations(input_img,annotations)
        path = "output/q1/test_lines/" + img_name
        save_img(test_lines_img,path)

        print("\n***************************************\n")
        print(CalcAngle(annotations[0], annotations[1], H))
        print(CalcAngle(annotations[2], annotations[3], H))
        print("\n***************************************\n")
        
        
    if args.question==2:
        H, h_affine = q2_metric_rectify(annotations)
        
        annotated_img = draw_annotations(input_img,annotations[-4:])
        annotated_img_affine = MyWarp(annotated_img, h_affine)

        path = "output/q2/annotated_img/" + img_name
        save_img(annotated_img,path)
        path = "output/q2/annotated_img_affine/" + img_name
        save_img(annotated_img_affine,path)

        result = MyWarp(input_img, H)
        path = "output/q2/rectified_image/" + img_name
        save_img(result,path)

        # Test line angles
        display_img(img_copy)
        test_lines_img = draw_annotations(input_img,annotations)
        path = "output/q2/test_lines/" + img_name
        save_img(test_lines_img,path)

        print("\n***************************************\n")
        print(CalcAngle(annotations[0], annotations[1], H))
        print(CalcAngle(annotations[2], annotations[3], H))
        print("\n***************************************\n")


    if args.question==3:
        
        H = compute_homography(annotations, img1)
        warped = cv2.warpPerspective(img1, H, (input_img.shape[1],input_img.shape[0]))
        
        blank_img = np.ones(img1.shape)
        blank_warped = cv2.warpPerspective(blank_img, H, (input_img.shape[1],input_img.shape[0]))
        mask = np.logical_not(blank_warped)

        result = warped + mask*input_img
        path = "output/q3/result/" + img_name
        save_img(result,path)

        for pt in annotations:
            cv2.circle(input_img, (pt[0], pt[1]), radius=10, color=(0,255,0), thickness=-1)
            cv2.circle(input_img, (pt[2], pt[3]), radius=10, color=(0,255,0), thickness=-1)
        # display_img(input_img)
        path = "output/q3/annotated_img/" + img_name
        save_img(input_img,path)


    if args.question==4:
        H = q4_metric_rectify(annotations)
        
        annotated_img = draw_annotations(input_img,annotations)

        path = "output/q4/annotated_img/" + img_name
        save_img(annotated_img,path)

        result = MyWarp(input_img, H)
        path = "output/q4/rectified_image/" + img_name
        save_img(result,path)

        # Test line angles
        display_img(img_copy)
        test_lines_img = draw_annotations(input_img,annotations)
        path = "output/q4/test_lines/" + img_name
        save_img(test_lines_img,path)

        test_lines_warped = MyWarp(test_lines_img, H)
        path = "output/q4/test_lines_warped/" + img_name
        save_img(test_lines_warped,path)

        print("\n***************************************\n")
        print(CalcAngle(annotations[0], annotations[1], H))
        print(CalcAngle(annotations[2], annotations[3], H))
        print("\n***************************************\n")



    if args.question==5:

        img_name = 'times_square.jpg'
        input_img = cv2.imread('./figures/TimesSquare.png', 1)
        img = cv2.imread('./data/my_images/federer.jpg', 1)
        img2 = cv2.imread('./data/my_images/federer-5.jpg', 1)
        img3 = cv2.imread('./data/my_images/federer-1.jpg', 1)
        
        imgs = [img, img2, img3]
        for img1 in imgs:
            img_copy = input_img.copy()
            display_img(img_copy)

            H = compute_homography(annotations, img1)
            warped = cv2.warpPerspective(img1, H, (input_img.shape[1],input_img.shape[0]))
            
            blank_img = np.ones(img1.shape)
            blank_warped = cv2.warpPerspective(blank_img, H, (input_img.shape[1],input_img.shape[0]))
            mask = np.logical_not(blank_warped)

            result = warped + mask*input_img
            input_img = result
        
        path = "output/q5/" + img_name
        # save_img(result,path)

    cv2.imshow('image', result)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    
    