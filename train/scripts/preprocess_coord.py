import argparse
import collections
import torch
import json
import numpy as np
import random
from interval import Interval
from tqdm import tqdm

# [Annotation type] [Object centric or Multiple instances] [Number of instances] [Number of keypoints] [Class A, Class B, ...] [Box A, Box B, ...]

# ----- kinhane
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

# ----- kinhane

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


def keypoint_to_formular_data(keypoints):
    output = []
    for kp_list in tqdm(keypoints):
        random.shuffle(kp_list)
        output_single = {"anno_type": "key point",
                         "prefix": "Multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": None,
                         "categories": [],
                         "coordinate": []
                         }
        for kp in kp_list:
            for name, point in kp.items():
                # ----- kinhane omit instances with less 3 key points
                if np.where(np.array(point)[:, -1] != 0)[0].shape[0] < 3:
                    continue
                # ----- kinhane
                output_single["instances_num"] += 1
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["keypoints_num"] = len(point)

                if output_single["instances_num"] > 7:
                    break

        # ----- kinhane omit idle list
        if len(output_single["coordinate"]) == 0:
            continue
        # ----- kinhane

        if random.random() < 0.5:
            flag = get_size(output_single["coordinate"], type='keypoint')  # add by kinhane
        else:
            flag = "random"
        output_single["flag"] = flag  # add by kinhane
        output.append(output_single)

    return output


def mask_to_formular_data(keypoints):
    output = []
    for mask_list in tqdm(keypoints):
        point_counter = 0

        random.shuffle(keypoints)
        output_single = {"anno_type": "mask",
                         "prefix": "Multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }
        for mask in mask_list:
            for name, point in mask.items():
                # ----- kinhane omit very small masks
                if len(point) < 5:
                    continue
                if point_counter + len(point) >= 150:
                    break
                else:
                    point_counter += len(point)

                # ----- kinhane
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["instances_num"] += 1



        # ----- kinhane omit idle list
        if len(output_single["coordinate"]) == 0:
            continue
        # ----- kinhane

        flag = get_size(output_single["coordinate"], type='mask')  # add by kinhane
        output_single["flag"] = flag  # add by kinhane
        output.append(output_single)

    return output


def box_to_formular_data(keypoints, centric=0):
    output = []
    for mask_list in tqdm(keypoints):
        random.shuffle(mask_list)
        output_single = {"anno_type": "box",
                         "prefix": "multiple instances",
                         "flag": None,
                         "instances_num": 0,
                         "keypoints_num": 0,
                         "categories": [],
                         "coordinate": []
                         }
        if centric == 1:
            output_single["prefix"] = "object centric"
        for mask in mask_list[:20]:
            for name, point in mask.items():
                output_single["categories"].append(name)
                output_single["coordinate"].append(point)
                output_single["instances_num"] += 1

        flag = get_size(output_single["coordinate"], type='box')  # add by kinhane
        output_single["flag"] = flag  # add by kinhane
        output.append(output_single)

    return output

num2char = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
            20: 'u', 21: 'v', 22: 'w'}

def formular_data_to_str(data_list, type):

    def keyporint_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + num2char[i] + ' $' + str(point[0]) + ' $'+ str(point[1])
            output = output + '] '
        return output

    def mask_coord_to_str(keypoints):
        output = ""
        for points_list in keypoints:
            output = output + '['
            for i, point in enumerate(points_list):
                output = output + ' ' + 'm'+str(i) + ' $' + str(point[0][0]) + ' $'+ str(point[0][1])
            output = output + '] '
        return output

    def box_coord_to_str(boxes):
        output = ""
        for box in boxes:
            output = output + '[ xmin $' + str(box[0]) + ' ymin $'+ str(box[1]) + \
                     ' xmax $'+ str(box[2]) + ' ymax $'+ str(box[3]) +'] '
        return output

    output = []
    for data in tqdm(data_list):
        output_single = '; '.join([data["anno_type"], data["prefix"], str(data["instances_num"]), str(data["keypoints_num"]), data['flag']])
        output_single = output_single + '; ' + ', '.join(data["categories"]) +'; '
        if type == "keypoint":
            output_single = output_single + keyporint_coord_to_str(data["coordinate"])
        elif type == "box":
            output_single = output_single + box_coord_to_str(data["coordinate"])
        else:
            output_single = output_single + mask_coord_to_str(data["coordinate"])
        output.append(output_single)

    return output

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, default='data/xjh_coco_val.json', help="data/xjh_coco_val.json")
    parser.add_argument("--output_path", type=str, default='test.txt', help="test.txt")
    parser.add_argument("--data_type", type=str, default='mask', help="box")
    parser.add_argument("--centric", type=int, default=0, help="box")

    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    print("data_type ", args.data_type)


    if args.data_type == "keypoint":
        keypoints = filter_keypoint(data['keypoints'] )
        data_json = keypoint_to_formular_data(keypoints)
    elif args.data_type == "box":
        data_json = box_to_formular_data(data['bboxes'], args.centric)
    else:
        data_json = mask_to_formular_data(data['masks'])

    data_str = formular_data_to_str(data_json, args.data_type)

    with open(args.output_path, 'w') as f:
        for l in data_str:
            f.write(l + '\n')


if __name__ == "__main__":
    main()