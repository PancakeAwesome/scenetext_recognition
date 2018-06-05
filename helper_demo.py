import numpy as np
import os
import shutil

def split_train_val(X, y, train_size):
    """Split dataset for training and validation.
        把数据集切分成训练集和验证集
    Args:
        X: A 1-D numpy array containing pathes of images.
        y: A 1-D numpy array containing labels.
        train_size: Size of training data to split.
    Returns:
        1-D numpy array having the same definition with X and y.
    """

    total_size = len(X)
    # 打乱数据
    shuffle_indices = np.random.permutation(np.arange(total_size))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    # 切分训练数据
    train_indices = np.random.choice(total_size, train_size, replace = False)
    X_train = X[train_indices]
    y_train = y[train_indices]

    # 切分验证集数据
    # 数据集其余的数据用来做验证集
    val_indices = [i for i in xrange(total_size) if i not in train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val

def write_to_file(data, file_to_output):
    """Write X_train/y_train/X_val/y_val/X_infer to file for further
       processing (e.g. make input queue of tensorflow).
    将训练集，验证集，测试集数据写入文件中
    Args:
        data: A 1-D numpy array, e.g, X_train/y_train/X_val/y_val/X_infer.
        file_to_output: A file to store data.
    """
    # with open('X_train.csv','a') as f_handle:
    #     np.savetxt(f_handle, X_train, fmt='%s', delimiter=",")

    with open(file_to_output, 'w') as f:
        for item in data.tolist():
            f.write(item + '\n')

# 从txt文件中读取labels
def load_labels(file):
    labels = list(open(file).readlines())
    # 去空格
    labels = [s.strip() for s in labels]
    labels = [s.split() for s in labels]

    labels_dict = dict(labels)

    labels = np.asarray(labels, dtype = str)
    labels = labels[:, 0]

    return labels, labels_dict

# 从文件夹中读取图片名字的列表
def load_img_path(images_path):
    tmp = os.listdir(images_path)
    tmp.sort(key = lambda x: int(x.split('.')[0]))

    file_names = [images_path + s for s in tmp]

    file_names = np.asarray(file_names)

    return file_names

def load_data(file_to_read):
    """Load X_train/y_train/X_val/y_val/X_infer for further
       processing (e.g. make input queue of tensorflow).

    Args:
        file_to_read:
    Returns:
        X_train/y_train/X_val/y_val/X_infer.
    """

    data = np.recfromtxt(file_to_read)
    data = np.asarray(data)

    return data


def cp_file(imgs_list_para, labels_list_para, dst_para):
    """
    最后的data格式为：./imgs/train/id_label.png
    """
    for i in xrange(imgs_list_para.shape[0]):
        file_path = imgs_list_para[i]

        filename = os.path.basename(file_path)
        fn = filename.split('.')[0]
        ext = filename.split('.')[1]

        dest_filename = dst_para + fn + '_' + labels_list_para[i] + '.' + ext

        shutil.copyfile(file_path, dest_filename)

if __name__ == '__main__':
    labels_path = './imgs/labels.txt'
    labels, labels_dict = load_labels(labels_path)
    # print(labels)

    images_path = './imgs/image_contest_level_1/'
    images_path_list = load_img_path(images_path)
    # print(images_path_list[:10])

    X_train, y_train, X_val, y_val = split_train_val(images_path_list, labels, 80000)
    write_to_file(X_train, "./imgs/X_train.txt")
    write_to_file(y_train, "./imgs/y_train.txt")
    write_to_file(X_val, "./imgs/X_val.txt")
    write_to_file(y_val, "./imgs/y_val.txt")

    cp_file(X_train, y_train, './imgs/train/')
    cp_file(X_val, y_val, './imgs/val/')