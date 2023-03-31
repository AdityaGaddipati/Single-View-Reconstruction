import numpy as np
import cv2
from utils import cvt_to_homogeneous, normalize

if __name__=="__main__":

    bunny_img = cv2.imread("data/q1/bunny.jpeg")
    twoD_3d_correspondences = np.loadtxt("data/q1/bunny.txt")
    # print(twoD_3d_correspondences)

    A = []
    for pts in twoD_3d_correspondences:

        x_2d = normalize(cvt_to_homogeneous([pts[0], pts[1]]))
        x_3d = normalize(cvt_to_homogeneous([pts[2], pts[3], pts[4]]))
        # x_3d = cvt_to_homogeneous([pts[2], pts[3], pts[4]])


        eq1 = np.hstack((np.zeros(4,), -x_2d[2]*x_3d, x_2d[1]*x_3d))
        eq2 = np.hstack((x_2d[2]*x_3d, np.zeros(4,), -x_2d[0]*x_3d))

        A.append(eq1)
        A.append(eq2)

    # A = np.array(A)
    # print(A)
    u, diag, v = np.linalg.svd(A)

    P = v[-1,:].reshape(3,4)
    print(P)

    # Surface points
    bunny_3d_pts = np.load("data/q1/bunny_pts.npy")
    bunny_3d_pts = np.hstack((bunny_3d_pts, np.ones((bunny_3d_pts.shape[0],1))))
    projected_pts = P.dot(bunny_3d_pts.transpose())

    projected_pts = (projected_pts/projected_pts[-1])[:2,:]
    for pts in projected_pts.transpose():
        cv2.circle(bunny_img, (int(pts[0]),int(pts[1])), radius=2, color=(0,0,255), thickness=-1)

    # Bounding Box
    # bunny_bbox = np.load("data/q1/bunny_bd.npy")
    # bunny_bbox = bunny_bbox.reshape((-1,3))

    # bunny_bbox = np.hstack( (bunny_bbox, np.ones((bunny_bbox.shape[0],1))) )
    # projected_pts = P.dot(bunny_bbox.transpose())

    # projected_pts = (projected_pts/projected_pts[-1])[:2,:]
    # for pts in projected_pts.transpose().reshape(-1,4):
    #     cv2.line (bunny_img, (int(pts[0]),int(pts[1])), (int(pts[2]),int(pts[3])),  color=(0,0,255), thickness=10) 

    
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow('image', bunny_img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    cv2.imwrite("./output/q1a_bunny.png", bunny_img)