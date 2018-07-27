from PIL import Image
import numpy as np
import json
import argparse
import glob
import os
import pdb

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--images_dir', type=str)
    aparser.add_argument('--ground_truth_dir', type=str)
    aparser.add_argument('--chip_size', type=int, default=400)
    aparser.add_argument('--overlap', type=int, default=0)
    aparser.add_argument('--output_dir', type=str, default=0)

    return aparser

def image_chipper(parent_img_name, chip_size, output_dir):
    """ Chips the full xview images and saves them into the output directory """
    img_arr = np.asarray(Image.open(parent_img_name))
    for i in range(0, round(img_arr.shape[0]/chip_size)):
        for j in range(0, round(img_arr.shape[1]/chip_size)):
            img_chip = img_arr[i*chip_size:i*chip_size+chip_size, j*chip_size:j*chip_size+chip_size, :]
            chip_name = '{}/{}_{}_{}.jpeg'.format(output_dir, os.path.splitext(parent_img_name.split('/')[-1])[0], i, j)
            img_chip = Image.fromarray(img_chip)
            img_chip.save(chip_name)
            yield chip_name, [i*chip_size+chip_size, j*chip_size+chip_size]

def ground_truth_parser(parent_img_name, chip_name, ground_truth, coords, chip_size=400):
    """ Parses the ground truth file for the dataset and finds the bounding box annotations
        in the chip of interest and saves them into a json file
    """
    chip_ground_truth = []
    offset = coords[0] - chip_size
    for dict_ann in ground_truth['features']:
        if dict_ann['properties']['image_id'] == parent_img_name.split('/')[-1]:
            parent_img_coords = dict_ann['properties']['bounds_imcoords'].split(',')
            if int(parent_img_coords[0]) > offset and int(parent_img_coords[1]) > offset:
                if int(parent_img_coords[2]) < coords[0] and int(parent_img_coords[3]) < coords[1]:
                    chip_ground_truth.append([[int(parent_img_coords[0]) - offset, int(parent_img_coords[1]) - offset,
                    int(parent_img_coords[2]) - offset, int(parent_img_coords[3])-offset], dict_ann['properties']['type_id']])

    with open('{}{}'.format(os.path.splitext(chip_name)[0], '.json'), 'w') as output_file:
        json.dump(chip_ground_truth, output_file)

def main():
    args = get_parser().parse_args()

    imgs_name = glob.glob('{}{}'.format(args.images_dir, '*.tif'))
    with open(args.ground_truth_dir) as f:
        ground_truth = json.load(f)

    for parent_img_name in imgs_name:
        chipper = image_chipper(parent_img_name, args.chip_size, args.output_dir)
        for i in chipper:
            chip_ground_truth = ground_truth_parser(parent_img_name, i[0], ground_truth, i[1], args.chip_size)

if __name__ == '__main__':
    main()
