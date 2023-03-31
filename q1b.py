import numpy as np
import cv2
import scipy.linalg
from utils import cvt_to_homogeneous, normalize

def getP(annotations, img):

    annotations = np.array(annotations).reshape(-1,2)

    pts_3d = np.array(([0,0,0],     # 1,2,3
        [1.27,0,0],
        [0,1.27,0],
        [0,0,1.27],
        [0,1.27,1.27],              # 2,3,6
        [1.27,0,1.27],              # 1,3,6
        [1.27,1.27,1.27]))

    print(pts_3d)

    A = []
    for i,pts in enumerate(annotations):

        x_2d = normalize(cvt_to_homogeneous([pts[0], pts[1]]))
        x_3d = normalize(cvt_to_homogeneous(pts_3d[i].tolist()))
        # x_3d = cvt_to_homogeneous([pts[2], pts[3], pts[4]])

        eq1 = np.hstack((np.zeros(4,), -x_2d[2]*x_3d, x_2d[1]*x_3d))
        eq2 = np.hstack((x_2d[2]*x_3d, np.zeros(4,), -x_2d[0]*x_3d))

        A.append(eq1)
        A.append(eq2)

    A = np.array(A)
    # print(A)
    u, diag, v = np.linalg.svd(A)

    P = v[-1,:].reshape(3,4)
    print("projection matrix")
    print(P)

    # R,Q = scipy.linalg.rq(P[:3,:3])
    # print("K matrix")
    # print(R/R[2,2])

    pts_3d = np.hstack((pts_3d, np.ones((len(pts_3d),1))))
    projected_pts = P.dot(pts_3d.T)

    projected_pts = (projected_pts/projected_pts[-1])[:2,:].astype(np.int).T

    lines = [[projected_pts[0], projected_pts[1]],
            [projected_pts[0], projected_pts[2]],
            [projected_pts[0], projected_pts[3]],
            [projected_pts[4], projected_pts[2]],
            [projected_pts[4], projected_pts[3]],
            [projected_pts[4], projected_pts[6]],
            [projected_pts[5], projected_pts[1]],
            [projected_pts[5], projected_pts[3]],
            [projected_pts[5], projected_pts[6]],
            ]

    # for pts in projesscted_pts.transpose():
    #     cv2.circle(bunny_img, (int(pts[0]),int(pts[1])), radius=2, color=(0,0,255), thickness=-1)

    for pts in lines:
        (x1,y1) = pts[0]
        (x2,y2) = pts[1]
        cv2.line (img, (x1,y1), (x2,y2),  color=(0,0,255), thickness=5) 

    return img
