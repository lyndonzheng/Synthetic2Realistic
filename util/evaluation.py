import argparse
from data_kitti import *

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--split', type=str, default='eigen', help='data split')
parser.add_argument('--predicted_depth_path', type=str, default='../dataset/KITTI31_predicted_lsgan/', help='path to estimated depth')
parser.add_argument('--gt_path', type = str, default='/data/dataset/KITTI/',
                    help = 'path to original kitti dataset /data/dataset/NYU_Test/testB')
parser.add_argument('--file_path', type = str, default='../datasplit/', help = 'path to datasplit files')
parser.add_argument('--save_path', type = str, default='/home/asus/lyndon/program/data/Image2Depth_31_KITTI/', help='path to save the train and test dataset')
parser.add_argument('--min_depth', type=float, default=1, help='minimun depth for evaluation')
parser.add_argument('--max_depth', type=float, default=50, help='maximun depth for evaluation, indoor 8.0, outdoor 50')
parser.add_argument('--normize_depth', type=float, default=80, help='depth normalization value, indoor 8.0, outdoor 80 (training scale)')
parser.add_argument('--eigen_crop',action='store_true', help='if set, crops according to Eigen NIPS14')
parser.add_argument('--garg_crop', action='store_true', help='if set, crops according to Garg  ECCV16')
args = parser.parse_args()

if __name__ == "__main__":

    predicted_depths = load_depth(args.predicted_depth_path,args.split, args.normize_depth)

    if args.split == 'indoor':

        ground_truths = load_depth(args.gt_path, args.split, 10)

        num_samples = len(ground_truths)

    elif args.split == 'eigen':
        test_files = read_text_lines(args.file_path + 'eigen_test_files.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)

        num_samples = len(im_files)
        ground_truths = []

        for t_id in range(num_samples):
            camera_id = cams[t_id]
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            ground_truths.append(depth.astype(np.float32))

            depth = cv2.resize(predicted_depths[t_id],(im_sizes[t_id][1], im_sizes[t_id][0]),interpolation=cv2.INTER_LINEAR)
            predicted_depths[t_id] = depth

    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples,np.float32)
    rmse = np.zeros(num_samples,np.float32)
    rmse_log = np.zeros(num_samples,np.float32)
    a1 = np.zeros(num_samples,np.float32)
    a2 = np.zeros(num_samples,np.float32)
    a3 = np.zeros(num_samples,np.float32)

    for i in range(len(ground_truths)):
    # for i in range(1):
        ground_depth = ground_truths[i]

        predicted_depth = predicted_depths[i]

        # print(ground_depth.max(),ground_depth.min())
        # print(predicted_depth.max(),predicted_depth.min())

        # depth_predicted = (predicted_depth / 7) * 255
        # depth_predicted = Image.fromarray(depth_predicted.astype(np.uint8))
        # depth_predicted.save(os.path.join('/home/asus/lyndon/program/Image2Depth/results/predicted_depth/', str(i)+'.png'))

        # depth = (depth / 80) * 255
        # depth = Image.fromarray(depth.astype(np.uint8))
        # depth.save(os.path.join('/data/result/syn_real_result/KITTI/ground_truth/{:05d}.png'.format(t_id)))

        predicted_depth[predicted_depth < args.min_depth] = args.min_depth
        predicted_depth[predicted_depth > args.max_depth] = args.max_depth

        if args.split == 'indoor':
            ground_depth = ground_depth[12:468, 16:624]

            height, width = ground_depth.shape
            predicted_depth = cv2.resize(predicted_depth,(width,height),interpolation=cv2.INTER_LINEAR)

            mask = np.logical_and(ground_depth > args.min_depth, ground_depth < args.max_depth)

        elif args.split == 'eigen':

            height, width = ground_depth.shape
            mask = np.logical_and(ground_depth > args.min_depth, ground_depth < args.max_depth)

            # crop used by Garg ECCV16
            if args.garg_crop:
                crop = np.array([0.40810811 * height,  0.99189189 * height,
                                     0.03594771 * width,   0.96405229 * width]).astype(np.int32)

            # crop we found by trail and error to reproduce Eigen NIPS14 results
            elif args.eigen_crop:
                crop = np.array([0.3324324 * height,  0.91351351 * height,
                                     0.0359477 * width,   0.96405229 * width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(ground_depth[mask],predicted_depth[mask])

        print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
              .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i]))

    print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
    print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
           .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))