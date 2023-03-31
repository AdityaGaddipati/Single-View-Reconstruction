import numpy as np
import scipy.linalg
from utils import get_line, crossProduct, cvt_to_inhomogeneous

def get_equation(pt1, pt2):
    return [pt1[0]*pt2[0] + pt1[1]*pt2[1], pt1[0]+pt2[0], pt1[1]+pt2[1], 1]
    # return [pt1[0]*pt2[0] + pt1[1]*pt2[1], pt1[0]*pt2[2]+pt1[2]*pt2[0], pt1[1]*pt2[2]+pt1[2]*pt2[1], pt1[2]*pt2[2]]


def getK_2a(annotations):

    l1 = get_line(annotations[0][:2], annotations[0][2:])
    m1 = get_line(annotations[1][:2], annotations[1][2:])
    
    l2 = get_line(annotations[2][:2], annotations[2][2:])
    m2 = get_line(annotations[3][:2], annotations[3][2:])

    l3 = get_line(annotations[4][:2], annotations[4][2:])
    m3 = get_line(annotations[5][:2], annotations[5][2:])

    vp1 = crossProduct(l1,m1)
    vp2 = crossProduct(l2,m2)
    vp3 = crossProduct(l3,m3)

    pt1 = cvt_to_inhomogeneous(vp1)
    pt2 = cvt_to_inhomogeneous(vp2)
    pt3 = cvt_to_inhomogeneous(vp3)
    
    print("Vanishing Pts")
    print(pt1)
    print(pt2)
    print(pt3)

    A = []
    A.append(get_equation(pt1,pt2))
    A.append(get_equation(pt2,pt3))
    A.append(get_equation(pt3,pt1))
    A = np.array(A)

    print("A matrix")
    print(A)

    u, diag, v = np.linalg.svd(A)

    omega = np.zeros((3,3))
    omega[0,0] = omega[1,1] = v[-1,0]
    omega[0,2] = omega[2,0] = v[-1,1]
    omega[1,2] = omega[2,1] = v[-1,2]
    omega[2,2] = v[-1,3]

    # omega = omega/omega[2,2]

    print("Image of absolute conic")
    print(omega)

    # print(np.linalg.eigvalsh(omega))

    K_invT = scipy.linalg.cholesky(omega, lower=False)
    K = np.linalg.inv(K_invT)
    K = K/K[2,2]

    print("K matrix")
    print(K)

    return K, [pt1,pt2,pt3]




