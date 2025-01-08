import open3d as o3d
import open3d.core as o3c
import glob
import numpy as np
import os


if __name__ == "__main__":
    point_dirs = r"D:\data\data\point_model\pointRcnn_data\points"
    save_dirs = r"D:\data\data\point_model\pointRcnn_data\downPoints"
    pts_file_paths = glob.glob(os.path.join(point_dirs, "*.npy"))
    x_min = 30
    y_min = 26
    z_min = 10
    x_max = -30
    y_max = -30
    z_max = -10
    for idx, pts_file_path in enumerate(pts_file_paths):
        # if idx > 0:
        #     continue
        print(idx, pts_file_path)
        data = np.load(pts_file_path)
        map_to_tensors = {}
        map_to_tensors["positions"] = o3c.Tensor(data[:, :3], o3c.float32)
        map_to_tensors["normals"] = o3c.Tensor(data[:, 3:], o3c.float32)
        pcd = o3d.t.geometry.PointCloud(map_to_tensors).to_legacy()
        
        x, y, z = pcd.get_min_bound()
        print("min: ", x, y, z)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if z < z_min:
            z_min = z
        x, y, z = pcd.get_max_bound()
        print("max: ", x, y, z)
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
        if z > z_max:
            z_max = z
        # # ---- down sample -----
        # downpcd_farthest = pcd.farthest_point_down_sample(20000)
        # dists = pcd.compute_point_cloud_distance(downpcd_farthest)
        # indexes = []
        # for i, d in enumerate(dists):
        #     if d > 0:
        #         continue
        #     indexes.append(i)
         
        # down_pcd = pcd.select_by_index(indexes)
         
        # pts = np.asarray(down_pcd.points)
        # normals = np.asarray(down_pcd.normals)
       
        # down_data = np.column_stack((pts, normals))
        # file_name = os.path.basename(pts_file_path)
        # np.save(os.path.join(save_dirs, file_name), down_data)
    print(x_min, y_min, z_min, x_max, y_max, z_max)
