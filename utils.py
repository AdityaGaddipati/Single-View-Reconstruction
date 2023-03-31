import numpy as np
import cv2

def normalize(v):
    # return v / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return v / np.sqrt(np.sum(v**2))

def MyWarp(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0,0],[0,h],[w,h],[w,0]], dtype=np.float64).reshape(-1,1,2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return result

def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2))


def cvt_to_homogeneous(x):
    '''
    Input:
    x: list of inhomogenous coordinates
    
    Output: homogenous coordinates np array
    '''
    # return(normalize(np.array([x[0],x[1],1])))
    x.append(1.0)
    return np.array(x)
    
def cvt_to_inhomogeneous(x):
    '''
    Input:
    x: homogeneous 3-vector
    
    Output: Inhomogenous vector
    '''
    return np.array([x[0]/x[2],x[1]/x[2]])
    
def crossProduct(a, b):
    '''
    Input:
    a,b: homogeneous 3-vectors
    
    Output: cross product of a,b
    '''
    return np.cross(a,b)


def get_line(pt1, pt2):
    '''
    Input:
    pt1, pt2: list [x,y] of points forming the line (inhomogeneous)
    
    Output: line 3-verctor
    '''
    p1 = cvt_to_homogeneous(pt1)
    p2 = cvt_to_homogeneous(pt2)
    return crossProduct(p1,p2)


def get_intersection(l1,l2):
    '''
    Input:
    l1, l2: line 3-vectors
    
    Output: intersection point 3-verctor
    '''
    return crossProduct(l1,l2)


def CalcAngle(l1, l2, H):
    '''
    Input:
    l1, l2: Input lines 
    H: Transform

    Output: Angle between lines
    '''

    l1 = get_line(l1)
    l2 = get_line(l2)

    l1_dash = np.linalg.inv(H).transpose().dot(l1)
    l2_dash = np.linalg.inv(H).transpose().dot(l2)

    return cosine(l1,l2), cosine(l1_dash,l2_dash)


def compute_homography(pts, h,w):

    '''
    Input:
        pts: Image points in numpy array
        h,w: target rectangle dimensions
    Output:
        H: homography between image points and target rectangle
    '''
    
    x1 = normalize(cvt_to_homogeneous([0,0]))
    x2 = normalize(cvt_to_homogeneous([w,0]))
    x3 = normalize(cvt_to_homogeneous([w,h]))
    x4 = normalize(cvt_to_homogeneous([0,h]))

    # x1 = normalize(cvt_to_homogeneous([0,h]))
    # x2 = normalize(cvt_to_homogeneous([w,h]))
    # x3 = normalize(cvt_to_homogeneous([w,0]))
    # x4 = normalize(cvt_to_homogeneous([0,0]))

    target_pts = [x1,x2,x3,x4]
    
    transformed_pts = []
    for pt in pts:
        transformed_pts.append(normalize(cvt_to_homogeneous(pt.tolist())))
        
    A = []
    for i,x in enumerate(target_pts):

        x_dash = transformed_pts[i]

        eq1 = np.hstack(([0,0,0], -x_dash[2]*x, x_dash[1]*x))
        eq2 = np.hstack((x_dash[2]*x, [0,0,0], -x_dash[0]*x))

        A.append(eq1)
        A.append(eq2)
        
    A = np.stack(A)
    
    u,diag,v = np.linalg.svd(A)
    
    H = v[-1,:].reshape(3,3)
    print(H)
    
    return H

def overlay_images(template, perpective_img, H):
    '''
    Input:
        template: template_img
        perpective_img: perpective_img
        H: Homography between them
    Output:
        result: template_img overlaid on perpective_img    
    '''

    warped = cv2.warpPerspective(template, H, (perpective_img.shape[1],perpective_img.shape[0]))
        
    blank_img = np.ones(template.shape)
    blank_warped = cv2.warpPerspective(blank_img, H, (perpective_img.shape[1],perpective_img.shape[0]))
    mask = np.logical_not(blank_warped)

    result = warped + mask*perpective_img

    return result