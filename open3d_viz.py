import numpy as np
import open3d as o3d
import pickle


if __name__=="__main__":

    # filename = 'q3b_1.pickle'
    # filename = 'q3b_2.pickle'
    filename = 'q3b_3.pickle'

    with open(filename, 'rb') as handle:
        pointcloud = pickle.load(handle)

    print(pointcloud.keys())
    xyz = pointcloud['points']
    rgb = pointcloud['color']

    print(xyz.shape)
    print(rgb.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd])