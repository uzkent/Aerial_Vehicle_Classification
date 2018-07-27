import numpy as np
from PIL import Image
import json
import pdb

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--images_dir', type=str)
    aparser.add_argument('--ground_truth_dir', type=str)
    aparser.add_argument('--chip_size', type=int, default=400)
    aparser.add_argument('--overlap', type=int, default=0)
    aparser.add_argument('--output_dir', type=int, default=0)

    return aparser

def image_chipper(img, chip_size, output_dir):

    img_arr = np.asarray(Image.load(img))
    for i in range(0, img_arr.width/chip_size):
        for j in range(0, img_arr.height/chip_size):
            img_chip = img_arr[i*chip_size:i*chip_size+chip_size, j*chip_size:j*chip_size+chip_size, :]
            img_chip.save('{}/{}_{}.jpeg'.format(output_dir, i, j)
            yield [i*chip_size+chip_size, j*chip_size+chip_size]

def ground_truth_parser(img, ground_truth, coords):
 
    for dict_ann in ground_truth['features']:
        if dict_ann['IMAGE_ID'] == img:
            if dict_ann['BOUNDS_IMCOORDS'][0] > coord[0] - chip_size and dict_ann['BOUNDS_IMCOORDS'][1] > coord[1] - chip_size:
                if dict_ann['BOUNDS_IMCOORDS'][2] < coord[0] and dict_ann['BOUNDS_IMCOORDS'][3] > coord[1]:
                    # [TODO] Use this annotation for the given chip

def main():
    args = get_parser().parse_args()

    imgs_name = glob.glob(args.images_dir + '*.tif')
    with open(args.ground_truth_dir) as f:
        ground_truth = json.load(f)

    for img in imgs_name:
        index, coords = image_chipper(img, args.chip_size, args.output_dir)
        ground_truth_parser(img, ground_truth, coords)
