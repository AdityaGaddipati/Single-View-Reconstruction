import numpy as np
import scipy.linalg
from utils import get_line, get_intersection, normalize, crossProduct, compute_homography

def getK_2b(annotations):

    H = []
    vanishing_pts = []
    for square in annotations:
        h_mat = compute_homography(square, 1,1)
        H.append(h_mat)

        l1 = get_line(square[0].tolist(), square[1].tolist())
        l2 = get_line(square[2].tolist(), square[3].tolist())
        vp = get_intersection(l1,l2)
        vanishing_pts.append(vp)

        m1 = get_line(square[0].tolist(), square[3].tolist())
        m2 = get_line(square[1].tolist(), square[2].tolist())
        vp = get_intersection(m1,m2)
        vanishing_pts.append(vp)


    A = []
    for h_mat in H:

        h1 = h_mat[:,0]
        h2 = h_mat[:,1]

        eq1 = np.hstack((h1*h2, [h1[0]*h2[1] + h1[1]*h2[0], h1[0]*h2[2] + h1[2]*h2[0], h1[1]*h2[2] + h1[2]*h2[1]]))

        tmp = np.array([h1[0]*h1[1] - h2[0]*h2[1], h1[0]*h1[2] - h2[0]*h2[2], h1[1]*h1[2] - h2[1]*h2[2]])
        eq2 = np.hstack((h1**2 - h2**2, 2*tmp))

        A.append(eq1)
        A.append(eq2)

    A = np.array(A)
    print(A)
    u, diag, v = np.linalg.svd(A)

    omega = np.eye(3)
    omega[0,0] = v[-1,0]
    omega[1,1] = v[-1,1]
    omega[2,2] = v[-1,2]
    omega[0,1] = omega[1,0] = v[-1,3]
    omega[0,2] = omega[2,0] = v[-1,4]
    omega[1,2] = omega[2,1] = v[-1,5]

    print("Image of absolute conic")
    print(omega)

    # print(np.linalg.eigvalsh(omega))

    K_invT = scipy.linalg.cholesky(omega, lower=False)
    K = np.linalg.inv(K_invT)
    K = K/K[2,2]

    print("K matrix")
    print(K)

    print("Vanishing Points")
    print(vanishing_pts)

    Kinv = np.linalg.inv(K)

    d1 = normalize(crossProduct(Kinv.dot(vanishing_pts[0]), Kinv.dot(vanishing_pts[1])))
    d2 = normalize(crossProduct(Kinv.dot(vanishing_pts[2]), Kinv.dot(vanishing_pts[3])))
    d3 = normalize(crossProduct(Kinv.dot(vanishing_pts[4]), Kinv.dot(vanishing_pts[5])))

    print(np.degrees(np.arccos(d1.dot(d2))))
    print(np.degrees(np.arccos(d2.dot(d3))))
    print(np.degrees(np.arccos(d3.dot(d1))))

