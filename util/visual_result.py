import matplotlib.pyplot as plt
import sys,os
sys.path.append('/home/asus/lyndon/program/Image2Depth/dataloader')
from dataloader.image_folder import make_dataset
import numpy as np
import scipy.misc

dataRoot = '/data/dataset/Image2Depth31_KITTI/testB'
dispairtyRoot = '/data/result/disparities_eigen_godard/disparities.npy'
depthSave = '/data/result/KITTI/Godard17'


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

# dataset, _ = make_dataset(dataRoot)
#
# for data in dataset:
#     output_name = os.path.splitext(os.path.basename(data))[0]
#     depth = ImageOps.invert(Image.open(data))
#     plt.imsave(os.path.join(depthSave, "{}.png".format(output_name)), depth, cmap='plasma')

dispairties = np.load(dispairtyRoot)
num, height, width = dispairties.shape
i = 0

for dispairty in dispairties:
    i += 1
    # dispairty = dispairty * dispairty.shape[1]
    # depth_pred = width_to_focal[1224] * 0.54 / dispairty
    # depth_pred[np.isinf(depth_pred)] = 0
    # depth_pred[depth_pred > 80] = 80
    # depth_pred[depth_pred < 1e-3] = 1e-3
    depth_to_img = scipy.misc.imresize(dispairty, [375, 1242])
    plt.imsave(os.path.join(depthSave, "{}_disp.png".format(i)), depth_to_img, cmap='plasma')
    # depth = width_to_focal[1224] * 0.54 / dispairty






