import numpy as np
import os
import cv2
from collections import Counter
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
from dataloader.image_folder import make_dataset

def compute_errors(ground_truth, predication):

    # accuracy
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_depth(file_path, split, max_depth):
    depths = []
    dataset, _ = make_dataset(file_path)
    for data in dataset:
        depth = Image.open(data)
        if split == 'outdoor':
            depth = np.array(depth)#[:,:,0]
            depth = depth.astype(np.float32) / 255 * max_depth
        elif split == 'indoor':
            depth = np.array(depth)#[:,:,0]
            depth = depth.astype(np.float32) / 255 * max_depth
            # depth = (depth - depth.mean()) / depth.std()
        depths.append(depth)
    return depths

def extract_train_file(args):

    if args.split == 'eigen':
        train_files = read_text_lines(args.file_path+'eigen_train_files.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(train_files,args.data_path)
        num_files = len(im_files)

        for t_id in range(num_files):
            image_name = '%010d.jpg' % (t_id)
            depth_path = os.path.join(args.save_path,'trainB/', image_name)
            image_path = os.path.join(args.save_path,'trainA/', image_name)

            image_name = ''
            camera_id = cams[t_id]
            depth, depth_interp = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, True, True)

            depth_img = Image.fromarray(np.uint8(depth_interp/80*255))
            depth_img.save(depth_path)

            rgb_img = cv2.imread(im_files[t_id])
            cv2.imwrite(image_path,rgb_img)

###############################################################################
#######################  EIGEN

def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines


def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0
    for filename in files:
        filename = filename.split()[0]
        splits = filename.split('/')
        camera_id = np.int32(splits[2][-1:])  # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        file_root = '{}/{}'

        im = filename
        vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)

        if os.path.isfile(data_root + im):
            gt_files.append(data_root + vel)
            gt_calib.append(data_root + date + '/')
            im_sizes.append(cv2.imread(data_root + im).shape[:2])
            im_files.append(data_root + im)
            cams.append(2)
        else:
            num_probs += 1
            print('{} missing'.format(data_root + im))
    print (num_probs, 'files missing')

    return gt_files, gt_calib, im_sizes, im_files, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P3_rect = cam2cam['P_rect_03'].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False):
    # load calibration files
    cam2cam = read_calib_file(calib_dir + 'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir + 'calib_velo_to_cam.txt')
    # print (velo2cam)
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth, depth_interp
    else:
        return depth