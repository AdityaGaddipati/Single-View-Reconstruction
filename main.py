import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import pickle

from q1b import getP
from q2a import getK_2a
from q2b import getK_2b
from q2c import getK_2c
from q3a import sv_reconstruct
from q3b import sv_reconstruct_3b

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
            cv2.line(img_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), thickness=5)
        
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

    # for pt in annotations:
    #     cv2.circle(viz_img, (pt[0],pt[1]), radius=5, color=(0,255,0), thickness=-1)

    return viz_img

def save_img(img, path):
    cv2.imwrite(path, img)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, help = "question number")
    args = parser.parse_args()

    #Q1b 
    if args.question=='1b':
        input_img = cv2.imread("data/dice2.jpg", 1)
        annotations = np.load("data/q1/q1b_.npy")
        print(annotations)
        # result=draw_annotations(input_img, annotations.reshape(-1,2))
        result = getP(annotations, input_img)

    # Q2a
    if args.question=='2a':
        input_img = cv2.imread("data/q2a.png", 1)
        # print(input_img.shape)
        annotations = np.load("data/q2/q2a.npy")
        # print(annotations.shape)
        annotations = annotations.reshape(-1,4).tolist()
        K, vp = getK_2a(annotations)

        # fig = plt.figure()
        # plt.imshow(input_img)
        # plt.plot([vp[0][0], vp[1][0]], [vp[0][1], vp[1][1]], color="blue", linewidth=3)
        # plt.plot([vp[1][0], vp[2][0]], [vp[1][1], vp[2][1]], color="blue", linewidth=3)
        # plt.plot([vp[0][0], vp[2][0]], [vp[0][1], vp[2][1]], color="blue", linewidth=3)
        # plt.plot(vp[0][0], vp[0][1],  marker='o', color="red")
        # plt.plot(vp[1][0], vp[1][1], marker='o', color="red")
        # plt.plot(vp[2][0], vp[2][1], marker='o', color="red")
        # plt.plot(int(K[0,2]), int(K[1,2]), marker='x', color="red")
        # fig.savefig('./output/q2a_2.png')
    
    
    # Q2b
    if args.question=='2b':
        annotations = np.load("data/q2/q2b.npy")
        print(annotations)
        K = getK_2b(annotations)

    # Q2c
    if args.question=='2c':
        annotations = np.load("data/q2/q2c.npy")
        K = getK_2c(annotations)

    #Q3a
    if args.question=='3a':
        input_img = cv2.imread("data/q3.png", 1)
        img_copy = input_img.copy()

        annotations = np.load("data/q3/q3.npy")
        print(annotations)
        sv_reconstruct(annotations, img_copy)

    #Q3b
    if args.question=='3b':
        # input_img = cv2.imread("data/q3b_1.jpeg", 1)
        # input_img = cv2.imread("data/q3b_2.jpeg", 1)
        input_img = cv2.imread("data/q3b_3.jpeg", 1)

        img_copy = input_img.copy()

        # display_img(img_copy)
        # annotations = np.array(annotations).reshape(2,-1,2)
        # annotations = np.array(annotations).reshape(5,-1,2)
        # annotations = np.array(annotations).reshape(3,-1,2)
        # np.save("./data/q3/q3b_2.npy",annotations)

        #q3b_1.jpeg
        # annotations = np.load("data/q3/q3b_1.npy")
        # print(annotations)
        # lines = []
        # lines.append(annotations[0][0].tolist() + annotations[0][1].tolist())
        # lines.append(annotations[0][2].tolist() + annotations[0][3].tolist())
        # lines.append(annotations[0][0].tolist() + annotations[0][3].tolist())
        # lines.append(annotations[0][1].tolist() + annotations[0][2].tolist())
        # lines.append(annotations[1][0].tolist() + annotations[1][3].tolist())
        # lines.append(annotations[1][1].tolist() + annotations[1][2].tolist())
        # plane_ref_points = [annotations[0][2].tolist(),annotations[0][2].tolist()]
        # K, vp = getK_2a(lines)

        #q3b_2.jpeg
        # annotations = np.load("data/q3/q3b_2.npy")
        # print(annotations)
        # lines = []
        # lines.append(annotations[3][0].tolist() + annotations[3][1].tolist())
        # lines.append(annotations[3][2].tolist() + annotations[3][3].tolist())
        # lines.append(annotations[3][0].tolist() + annotations[3][3].tolist())
        # lines.append(annotations[3][1].tolist() + annotations[3][2].tolist())
        # lines.append(annotations[4][0].tolist() + annotations[4][3].tolist())
        # lines.append(annotations[4][1].tolist() + annotations[4][2].tolist())
        # plane_ref_points = [annotations[0][3].tolist(),annotations[0][3].tolist(),annotations[2][3].tolist(),annotations[0][3].tolist(),annotations[2][3].tolist()]
        # K, vp = getK_2a(lines)

        # #q3b_3.jpeg
        annotations = np.load("data/q3/q3b_3.npy")
        print(annotations)
        lines = []
        lines.append(annotations[0][0].tolist() + annotations[0][1].tolist())
        lines.append(annotations[0][2].tolist() + annotations[0][3].tolist())
        lines.append(annotations[1][0].tolist() + annotations[1][1].tolist())
        lines.append(annotations[1][2].tolist() + annotations[1][3].tolist())
        lines.append(annotations[2][0].tolist() + annotations[2][3].tolist())
        lines.append(annotations[2][1].tolist() + annotations[2][2].tolist())
        plane_ref_points = [annotations[0][1].tolist(),annotations[0][1].tolist(),annotations[0][1].tolist()]
        K, vp = getK_2a(lines)

        points_3d, color = sv_reconstruct_3b(annotations, img_copy, K, plane_ref_points)
        print(points_3d.shape, color.shape)

        pointcloud={}
        pointcloud['points'] = points_3d
        pointcloud['color'] = color[:,:3]

        with open('q3b_3.pickle', 'wb') as handle:
            pickle.dump(pointcloud, handle, protocol=pickle.DEFAULT_PROTOCOL)
