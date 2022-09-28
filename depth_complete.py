import os
import glob
import cv2
import sys
import imageio
import argparse
import numpy as np
from pathlib import Path

from utils.depth_util import vel2cam, fill_in_fast, fill_in_multiscale

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def complete(opt):
    """Depth maps are saved to the 'outputs' folder.
    """
    fill_type,  blur_type = opt.fill_type, opt.blur_type

    # the folder set
    folder = Path(ROOT / opt.dataset / opt.name)
    img_folder = Path(folder / f'image_2/')
    velo_folder = Path(folder / f'velodyne/')
    cali_folder = Path(folder / f'calib/')
    save_folder = Path(folder / f'depth/')
    comp_folder = Path(folder / f'depth_val/')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(comp_folder, exist_ok=True)

    # Get images in sorted order
    img_list = sorted(glob.glob(str(img_folder / '*.png')))
    num = len(img_list)
    for i in range(num):
        # check the file
        img_file = img_list[i]
        img_index = os.path.split(img_file)[1].split('.')[0]
        velo_file = str(velo_folder / f'{img_index}.bin')
        cali_file = str(cali_folder / f'{img_index}.txt')
        save_file = str(save_folder / f'{img_index}.png')
        if os.path.exists(velo_file) and os.path.exists(cali_file):
            # to convert velo to png
            depth = vel2cam(img_file, velo_file, cali_file, save_file)
            # Show progress
            sys.stdout.write('\rProcessing {} / {} \n'.format(i + 1, num))
            sys.stdout.flush()
        else:
            print(f"{img_index}.bin or {img_index}.txt miss! Please check!\n")
            continue

        # Fill in
        depth = np.float32(depth / 256.0)
        if fill_type == 'fast':
            depth_filed = fill_in_fast(
                depth, extrapolate=False, blur_type=blur_type)
        elif fill_type == 'multiscale':
            depth_filed, process_dict = fill_in_multiscale(
                depth, extrapolate=False, blur_type=blur_type,
                show_process=opt.show_process)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))

        # Display images from process_dict
        if fill_type == 'multiscale' and opt.show_process:
            img_size = (570, 165)

            x_start = 80
            x_offset = img_size[0]
            x_padding = 0

            img_x = x_start
            max_x = 1900

            row_idx = 0
            for key, value in process_dict.items():
                image_jet = cv2.applyColorMap(
                    np.uint8(value / np.amax(value) * 255),
                    cv2.COLORMAP_JET)

                img_x += x_offset + x_padding
                if (img_x + x_offset + x_padding) > max_x:
                    img_x = x_start
                    row_idx += 1

                # Save process images
                cv2.imwrite('process/' + key + '.png', image_jet)

        # Save depth map to uint16 png (same format as disparity maps)
        comp_file = str(comp_folder / f'{img_index}.png')
        imageio.imwrite(comp_file, depth_filed.astype(np.uint16))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--name', type=str, default='training')
    parser.add_argument('--fill-type', choices=['fast', 'multiscale'], type=str, default='fast')
    parser.add_argument('--blur-type', choices=['gaussian', 'bilateral'], type=str, default='bilateral')
    parser.add_argument('--show-process', type=bool, default=False, help='show process in multiscale')
    opt = parser.parse_args()
    return opt


def main(**kwargs):
    opt = parse_opt()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    complete(opt)

if __name__ == "__main__":
    main()
