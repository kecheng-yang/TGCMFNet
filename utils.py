# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 13:26
@Author ：Kexin Ding
@FileName ：utils.py
"""
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from scipy.io import loadmat
import random


def get_dataset(dataset, num_labelled):
    if dataset == "houston2013":
        DataPath1 = r'./datasets/houston2013/HSI_data.mat'
        DataPath2 = r'./datasets/houston2013/LiDAR_data.mat'
        Data1 = loadmat(DataPath1)['HSI_data']  # (349,1905,144)
        Data2 = loadmat(DataPath2)['LiDAR_data']
        LabelPath = r'./datasets/houston2013/All_Label.mat'
        gt = loadmat(LabelPath)['All_Label']
        num_class = gt.max()
        TrLabel, TsLabel = data_partition(gt, num_class, train_num=num_labelled)
        label_values = ["Health grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential", "Commercial",
                        "Road", "Highway", "Railway", "Parking lot 1", "Parking lot 2", "Tennis court", "Running track"]


        # tongyi
        label_queue = {
                    "Health grass":     ["Healthy grass is recognized by its vibrant green color, indicating robust growth and well-maintained condition.",
                                        "Healthy grass usually flourishes near water bodies and trees, thriving in well-watered and sunny areas."],
                    "Stressed grass":   ["Stressed grass appears less vivid or yellowish compared to healthy grass, indicating poor health or lack of water.",
                                        "Stressed grass can be found in areas with insufficient water supply or poor soil conditions."],
                    "Synthetic grass":  ["Synthetic grass has a uniform green color without variation, which makes it distinguishable from natural grass.",
                                        "Synthetic grass is typically located in sports fields or playgrounds where maintenance of natural grass could be challenging."],
                    "Trees":            ["Trees are characterized by their darker green color and shadowing effects in high-resolution images, reflecting dense foliage.",
                                        "Trees are often grouped together forming clusters within parks, residential areas, or forests."],
                    "Soil":             ["Soil exhibits various colors ranging from light brown to dark brown depending on its composition and moisture content.",
                                        "Soil is generally found in open areas not covered by vegetation, such as construction sites or barren land."],
                    "Water":            ["Water bodies appear dark blue or black in the high spectral range due to absorption of light, making them easily identifiable.",
                                        "Water bodies have a smooth surface and can be lakes, ponds, rivers, or even swimming pools."],
                    "Residential":      ["Residential areas show a mix of different materials like roofs, roads, and vegetation, giving a diverse spectral signature.",
                                        "Residential areas are organized into blocks and grids, often interspersed with trees and small gardens."],
                    "Commercial":       ["Commercial areas feature large buildings, parking lots, and less greenery, resulting in distinct spectral characteristics.",
                                        "Commercial areas tend to be more centralized and clustered around major roadways or intersections."],
                    "Road":             ["Roads are distinguished by their gray or white color, depending on the material used, and are clearly defined linear structures.",
                                        "Roads connect different parts of the urban area and are often lined with trees or buildings."],
                    "Highway":          ["Highways are wider than regular roads and may include multiple lanes, showing up as broader lines on the map.",
                                        "Highways are designed for high-speed traffic and are often accompanied by sound barriers or green belts."],
                    "Railway":          ["Railways are long, narrow strips that contrast sharply with surrounding areas, often appearing as dark lines.",
                                        "Railways run through both urban and rural landscapes, sometimes elevated or below ground level."],
                    "Parking lot 1":    ["Parking lot 1 is characterized by its flat, light-colored surfaces with minimal vegetation, offering clear spectral signatures.",
                                        "Parking lot 1 is usually adjacent to commercial or industrial zones, providing temporary vehicle storage space."],
                    "Parking lot 2":    ["Parking lot 2 may have different surface treatments or materials compared to Parking lot 1, leading to slight variations in spectral reflectance.",
                                        "Parking lot 2 serves as a parking area but might be located in different settings or serve different purposes."],
                    "Tennis court":     ["Tennis courts have a very specific layout and color scheme, usually featuring green or red surfaces with white lines.",
                                        "Tennis courts are typically found within recreational areas or sports complexes, surrounded by fencing."],
                    "Running track":    ["Running track is typically a well-defined oval shape, usually with a red or other colored surface and white lines for lanes.",
                                        "Running track is commonly found in sports complexes or parks, often near other sports facilities like football fields or tennis courts."]
                    }

    elif dataset == "muufl":
        DataPath1 = r'./datasets/muufl/HSI_data.mat'
        DataPath2 = r'./datasets/muufl/LiDAR_data.mat'
        Data1 = loadmat(DataPath1)['HSI_data']
        Data2 = loadmat(DataPath2)['LiDAR_data']
        LabelPath = r'./datasets/muufl/All_Label.mat'
        gt = loadmat(LabelPath)['All_Label']
        num_class = gt.max()
        TrLabel, TsLabel = data_partition(loadmat(LabelPath)['All_Label'], num_class, train_num=num_labelled)
        label_values = None
        # gpt
        label_queue = None


    elif dataset == "trento":
        DataPath1 = r'./datasets/trento/HSI_data.mat'
        DataPath2 = r'./datasets/trento/LiDAR_data.mat'
        Data1 = loadmat(DataPath1)['HSI_data']
        Data2 = loadmat(DataPath2)['LiDAR_data']
        LabelPath = r'./datasets/trento/All_Label.mat'
        gt = loadmat(LabelPath)['All_Label']
        num_class = gt.max()
        TrLabel, TsLabel = data_partition(loadmat(LabelPath)['All_Label'], num_class, train_num=num_labelled)
        label_values = None
        # kimi
        label_queue = None

    elif dataset == "augsburg":
        DataPath1 = r'./datasets/augsburg/HSI_data.mat'
        DataPath2 = r'./datasets/augsburg/LiDAR_data.mat'
        Data1 = loadmat(DataPath1)['HSI_data']
        Data2 = loadmat(DataPath2)['LiDAR_data']
        LabelPath = r'./datasets/augsburg/All_Label.mat'
        gt = loadmat(LabelPath)['All_Label']
        num_class = gt.max()
        TrLabel, TsLabel = data_partition(loadmat(LabelPath)['All_Label'], num_class, train_num=num_labelled)
        label_values = None
        # kimi
        label_queue = None

    elif dataset == "berlin":
        DataPath1 = r'./datasets/berlin/HSI_data.mat'
        DataPath2 = r'./datasets/berlin/LiDAR_data.mat'
        Data1 = loadmat(DataPath1)['HSI_data']
        Data2 = loadmat(DataPath2)['LiDAR_data']
        LabelPath = r'./datasets/berlin/All_Label.mat'
        gt = loadmat(LabelPath)['All_Label']
        num_class = gt.max()
        TrLabel, TsLabel = data_partition(loadmat(LabelPath)['All_Label'], num_class, train_num=num_labelled)
        label_values = None
        label_queue = None
    else:
        Data1=None
        Data2=None
        TrLabel=None
        TsLabel=None
        num_class=None
        label_values = None
        label_queue = None
        print("No such dataset")
        import sys
        sys.exit()

    return Data1, Data2, TrLabel, TsLabel, num_class, label_values, label_queue

def data_partition(gt, class_count, train_num=20):
    gt_reshape = np.reshape(gt, [-1])  #gt展开

    train_idx = np.array([]).astype('int64')    #存储训练集序号
    test_idx = np.array([]).astype('int64')     #存储测试集序号

    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1] #获取标签为i+1的类别数量

        np.random.shuffle(idx)
        samplesCount = len(idx) #标签为i+1样本的数量
        if train_num < samplesCount:
            train_simples_idx = idx[:train_num]
            test_simples_idx = idx[train_num:]
        else:
            train_simples_idx = idx
            test_simples_idx = idx[samplesCount:]
        train_idx = np.concatenate((train_idx, train_simples_idx), axis=0)

        test_idx = np.concatenate((test_idx, test_simples_idx), axis=0)
    train_gt = np.zeros(gt_reshape.shape)
    test_gt = np.zeros(gt_reshape.shape)
    train_gt[train_idx] = gt_reshape[train_idx]
    test_gt[test_idx] = gt_reshape[test_idx]
    train_gt = train_gt.reshape(gt.shape)
    test_gt = test_gt.reshape(gt.shape)

    return train_gt, test_gt


def train_patch(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)


    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')


    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width


    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()

    return TrainPatch1, TrainPatch2, TrainLabel


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = torch.sum(predict==label)*1.0/n
    correct_sum = torch.zeros((max(label)+1))
    reali = torch.zeros((max(label)+1))
    predicti = torch.zeros((max(label)+1))
    CA = torch.zeros((max(label)+1))
    for i in range(0, max(label) + 1):
        correct_sum[i] = torch.sum(label[np.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
   val_predict_labels = torch.squeeze(val_predict_labels)
   val_true_labels = torch.squeeze(val_true_labels)
   OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
   # ic(OA, Kappa, CA, AA)
   print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
   print("CA: ",)
   print(CA)
   return OA, Kappa, CA, AA

