import numpy as np
import vedo 


def show_pts(pts_points):
    colors = ["grey", "red", "blue", "green", "yellow", "black", "pink", "white", "brown", "orange", "purple", "cyan",
              "teal", "magenta", "gold", "silver", "beige", "marron", "amber", "navy", "ruby"]
    sizes = [4, 4, 4, 4]

    show_pts_point = []
    for idx, pts_point in enumerate(pts_points):
        point = vedo.Points(pts_point.reshape(-1, 3)).c((colors[idx % len(colors)]))
        show_pts_point.append(point)
 
    vedo.show(show_pts_point)


if __name__ == "__main__":
    npy_file = "D:/test.npy"
    data = np.load(npy_file)
    labels = data[:, -1]  # 获取标签列
 
    unique_labels = np.unique(labels)
 
    grouped = [data[labels == label][:, :3] for label in unique_labels]
    
    show_pts(grouped)
