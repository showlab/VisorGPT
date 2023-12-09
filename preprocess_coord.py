import argparse
import json
import numpy as np
import random
from interval import Interval
from tqdm import tqdm
import os

def get_size(coordinate, type, small=Interval(0, 32**2), medium=Interval(32**2, 96**2, lower_closed=False)):

    if type == 'box':
        coordinate = np.array(coordinate)
        mean_area = np.mean((coordinate[:, 2] - coordinate[:, 0]) * (coordinate[:, 3] - coordinate[:, 1]))
        # import ipdb
        # ipdb.set_trace()
    elif type == 'keypoint' or type == 'mask':
        area_list = []
        for coord in coordinate:
            if type == 'mask':
                coord = np.array(coord).squeeze(1)
            else:
                # delete unannotated key points
                tmp = []
                for kpt in coord:
                    _, _, v = kpt
                    if v != 0:
                        tmp.append(kpt)
                coord = np.array(tmp)
            # import ipdb
            # ipdb.set_trace()
            area = (np.max(coord[:, 0]) - np.min(coord[:, 0])) * (np.max(coord[:, 1]) - np.min(coord[:, 1]))
            area_list.append(area)
        mean_area = np.mean(area_list)
    else:
        raise NotImplementedError

    if mean_area in small:
        return 'small'
    elif mean_area in medium:
        return 'medium'
    else:
        return 'large'

def filter_keypoint(keypoints):
    output = []
    for kp_list in keypoints:
        output_single = []
        for kp in kp_list:
            for name, point in kp.items():
                if np.array(point).sum() > 0:
                    output_single.append({name: point})
        if len(output_single) > 0:
            output.append(output_single)
    return output

def sample_data(data, num_samples):
    # compute the number of samples
    num_epochs = num_samples // len(data)
    if num_samples != len(data):
        num_epochs += 1

    data *= num_epochs
    print(num_epochs, len(data))
    data = random.sample(data, num_samples)

    return data

def keypoint_to_formular_data(data, num_samples=-1):

    if num_samples != -1:
        data = sample_data(data, num_samples)

    output = []
    for kp_list in tqdm(data):
        random.shuffle(kp_list)
        output_single = {"anno_type": "key point",
                         "prefix": "multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": None,
                         "categories": [],
                         "coordinate": []
                         }
        for kp in kp_list:
            for name, point in kp.items():
                # ----- kinhane omit instances with less 5 key points
                if np.where(np.array(point)[:, -1] != 0)[0].shape[0] <= 6:
                    continue
                # ----- kinhane
                output_single["instances_num"] += 1
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["keypoints_num"] = len(point)

            # if output_single["instances_num"] > 7:
            if output_single["instances_num"] > 14:
                break

        # ----- kinhane omit idle list
        if len(output_single["coordinate"]) == 0:
            continue

        flag = get_size(output_single["coordinate"], type='keypoint')  # add by kinhane

        output_single["flag"] = flag  # add by kinhane
        output.append(output_single)

    return output


def mask_to_formular_data(data, num_samples=-1):

    if num_samples != -1:
        data = sample_data(data, num_samples)

    output = []
    for mask_list in tqdm(data):
        random.shuffle(mask_list)
        output_single = {"anno_type": "mask",
                         "prefix": "multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }

        for mask in mask_list:
            for name, point in mask.items():
                point = point['coords']

                # ----- kinhane omit very small masks
                if len(point) < 5:
                    continue
                # ----- kinhane
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["instances_num"] += 1

            if output_single["instances_num"] > 7:  # 36 points each mask
                break

        if len(output_single["coordinate"]) == 0:
            continue

        # ----- kinhane
        flag = get_size(output_single["coordinate"], type='mask')  # add by kinhane

        output_single["flag"] = flag
        output.append(output_single)

    return output


def box_to_formular_data(data, centric=0, num_samples=-1, anno_type='box'):

    if num_samples != -1:
        data = sample_data(data, num_samples)

    output = []
    for mask_list in tqdm(data):
        random.shuffle(mask_list)
        output_single = {"anno_type": anno_type,
                         "prefix": "multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }
        if centric == 1:
            output_single["prefix"] = "object centric"

        for mask in mask_list:
            for name, point in mask.items():
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["instances_num"] += 1

        if len(output_single["coordinate"]) == 0:
            continue

        flag = get_size(output_single["coordinate"], type='box')  # add by kinhane

        output_single["flag"] = flag
        output.append(output_single)

    return output

num2char = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
            20: 'u', 21: 'v', 22: 'w'}

def formular_data_to_str(data_list, type):

    def keypoint_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + num2char[i] + ' ' + str(point[0]) + ' '+ str(point[1])
            output = output + '] '
        return output

    def mask_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + 'm'+str(i) + ' ' + str(point[0][0]) + ' '+ str(point[0][1])
            output = output + '] '
        return output

    def box_coord_to_str(boxes):
        output = ""
        for box in boxes:
            output = output + '[ xmin ' + str(box[0]) + ' ymin '+ str(box[1]) + \
                     ' xmax '+ str(box[2]) + ' ymax '+ str(box[3]) +'] '
        return output

    output = []
    for data in tqdm(data_list):

        if random.random() > 0.5:
            # prompt 1
            output_single = '; '.join([data["anno_type"], data["prefix"], data["flag"], str(data["instances_num"]), str(data["keypoints_num"])])

            output_single = output_single + '; ' + ', '.join(data["categories"]) + '; '
            if type == "keypoint":
                output_single = output_single + keypoint_coord_to_str(data["coordinate"])
            elif type == "box":
                output_single = output_single + box_coord_to_str(data["coordinate"])
            else:
                output_single = output_single + mask_coord_to_str(data["coordinate"])
            output.append(output_single)
        else:
            # prompt 2
            output_single = '; '.join([data["anno_type"], data["prefix"], data["flag"], str(data["instances_num"]), str(data["keypoints_num"])]) + '; '
            if type == "keypoint":
                str_coord = keypoint_coord_to_str(data["coordinate"])
            elif type == "box":
                str_coord = box_coord_to_str(data["coordinate"])
            else:
                str_coord = mask_coord_to_str(data["coordinate"])

            str_coord = str_coord.replace('[ ', '').split('] ')[:-1]

            for cat, coord in zip(data['categories'], str_coord):
                output_single = output_single + '[ ' + cat + ' ' + coord + ' ] '

            output.append(output_single)

    return output

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, default='data/coco_val.json', help="path to your json file")
    parser.add_argument("--output_dir", type=str, default='txt_train', help="directory to store .txt files")
    parser.add_argument("--data_type", type=str, default='mask', help="annotation type")
    parser.add_argument("--centric", type=int, default=0, help="0 for imagenet")
    parser.add_argument("--num_samples", type=int, default=-1, help="how many samples, -1 means no oversampling")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input_path) as f:
        data = json.load(f)

    print("data_type ", args.data_type)

    if args.data_type == "keypoint":
        keypoints = filter_keypoint(data['keypoints'])
        data_json = keypoint_to_formular_data(keypoints, args.num_samples)
    elif args.data_type == "box":
        if args.num_samples == -1:
            args.num_samples = len(data['bboxes'])
        data = data['bboxes']

        if 'rico' in args.input_path:
            data_json = box_to_formular_data(data, args.centric, args.num_samples, anno_type='rico')
        elif 'publaynet' in args.input_path:
            data_json = box_to_formular_data(data, args.centric, args.num_samples, anno_type='publaynet')
        else:
            data_json = box_to_formular_data(data, args.centric, args.num_samples)
    else:
        if args.num_samples == -1:
            args.num_samples = len(data['bboxes'])
        data = data['masks']
        data_json = mask_to_formular_data(data, args.num_samples)

    data_str = formular_data_to_str(data_json, args.data_type)

    save_path = args.input_path.split('/')[-1].split('.')[0] + f'_{args.data_type}.txt'
    with open(os.path.join(args.output_dir, save_path), 'w') as f:
        for l in data_str:
            f.write(l + '\n')


if __name__ == "__main__":
    main()