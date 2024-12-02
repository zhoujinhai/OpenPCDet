import numpy as np
import glob
import os


# 定义一个转换函数，将浮点数限定为六位小数
def limit_float_digits(s):
    val = float(s)
    return float(f"{val:.6f}")


if __name__ == "__main__":
    label_paths_dir = r"D:\data\data\point_model\pt_and_label"
    save_dir = r"D:\data\data\point_model\pointRcnn_data"

    label_paths = glob.glob(os.path.join(label_paths_dir, "*.pts"))
    image_sets_dir = os.path.join(save_dir, "ImageSets")
    labels_dir = os.path.join(save_dir, "labels")
    points_dir = os.path.join(save_dir, "points")
    if not os.path.exists(image_sets_dir):
        os.mkdir(image_sets_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
    if not os.path.exists(points_dir):
        os.mkdir(points_dir)

    val_ratio = 0.2
    val_n = int(len(label_paths) * val_ratio)
    train_n = int(len(label_paths) - val_n)
    print("val: ", val_n, "train: ", train_n)
    train_idx = []
    val_idx = []
    random_idx = np.concatenate((np.zeros(val_n, int), np.ones(train_n, int)))
    np.random.shuffle(random_idx)
    for idx, label_path in enumerate(label_paths):
        file_name = os.path.splitext(os.path.basename(label_path))[0]
        print(idx, file_name)
        if random_idx[idx]:
            train_idx.append(file_name)
        else:
            val_idx.append(file_name)
        data = np.loadtxt(label_path, converters={0: limit_float_digits})
        points = data[:, 0:6]
        np.save(os.path.join(points_dir, file_name + ".npy"), points)
        number = int(np.max(data[:, -1]))
        with open(os.path.join(labels_dir, file_name + ".pts"), "w") as f:
            for i in range(number + 1):
                box = data[data[:, -1] == i][0, 6:13]
                str_box = ['{:.6f}'.format(x) for x in box]
                for s in str_box:
                    f.write(s + " ")
                f.write("teeth" + "\n")
    np.savetxt(os.path.join(image_sets_dir, "train.pts"), np.asarray(train_idx), fmt
