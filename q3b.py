import numpy as np
import cv2
import matplotlib.pyplot as plt
from q2a import getK_2a
from utils import get_line, get_intersection, normalize, crossProduct

def reproject_plane(img, K, polygon, normal, point):
        mask = np.zeros(img.shape)

        mask = cv2.fillPoly(mask, [polygon], color=(255, 255, 255))
        mask = mask>1

        plane_img = mask*img
        pixel_sum = plane_img.sum(axis=2)
        non_black = pixel_sum != 0
        y_index, x_index = np.where(non_black)
        print(x_index.shape)

        # y_index = y_index[np.arange(0,x_index.shape[0],10)]
        # x_index = x_index[np.arange(0,x_index.shape[0],10)]
        # print(x_index.shape)
        
        rgba_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        rgba = rgba_img[y_index,x_index,:]/255

        Kinv = np.linalg.inv(K)
        c = -normal.dot(Kinv.dot(np.array(point)))

        homo_pts = np.stack((x_index,y_index,np.ones(x_index.shape[0],)))

        rays = Kinv.dot(homo_pts)
        mu = -c/normal.dot(rays)

        pts_3d = mu*rays

        return pts_3d, rgba


def sv_reconstruct_3b(annotations, img, K, plane_ref_points):

    Kinv = np.linalg.inv(K)

    normals = []
    for plane in annotations:
        l1 = get_line(plane[0].tolist(), plane[1].tolist())
        l2 = get_line(plane[2].tolist(), plane[3].tolist())
        vp1 = get_intersection(l1,l2)

        m1 = get_line(plane[0].tolist(), plane[3].tolist())
        m2 = get_line(plane[1].tolist(), plane[2].tolist())
        vp2 = get_intersection(m1,m2)

        n1 = normalize(crossProduct(Kinv.dot(vp1), Kinv.dot(vp2)))
        normals.append(n1)

    print(np.degrees(np.arccos(normals[0].dot(normals[1]))))   

    points_3d, color = [], []   
    for i,pts in enumerate(plane_ref_points):
        pts_3d, rgba = reproject_plane(img, K, annotations[i], normals[i], [pts[0],pts[1],1])

        if len(points_3d) == 0:
            points_3d = pts_3d.T
            color = rgba
        else:
            points_3d = np.vstack((points_3d, pts_3d.T))
            color = np.vstack((color, rgba))

    return points_3d, color
