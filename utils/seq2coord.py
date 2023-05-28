
"""
decode sequential output to visual locations
author: sierkinhane.github.io
"""
import random
from tqdm import tqdm
import json
import numpy as np
import re
import argparse
import cv2
import math
import os

# COCO keypoints
stickwidth = 4

limbSeq_coco = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

limbSeq_cp = [[14, 2], [14, 1], [2, 4], [4, 6], [1, 3], [3, 5], [14, 8], [8, 10], [10, 12], [14, 7], [7, 9], [9, 11], [13, 14]]

# CrowdPose
# {'0': 'left shoulder', '1': 'right shoulder', '2': 'left elbow', '3': 'right elbow', '4': 'left wrist', '5': 'right wrist', '6': 'left hip', '7': 'right hip', '8': 'left knee', '9': 'right knee', '10': 'left ankle', '11': 'right ankle', '12': 'head', '13': 'neck'}

# for human pose visualization
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# for box visualization
colors_box = [[217, 221, 116], [137, 165, 171], [230, 126, 175], [63, 157, 5], [107, 51, 75], [217, 147, 152], [129, 132, 8], [232, 85, 249], [254, 98, 33], [89, 108, 230], [253, 34, 161], [91, 150, 30], [255, 147, 26], [209, 154, 205], [134, 57, 11], [143, 181, 122], [241, 176, 87], [104, 73, 26], [122, 147, 59], [235, 230, 229], [119, 18, 125], [185, 61, 138], [237, 115, 90], [13, 209, 111], [219, 172, 212]]

# Plots one bounding box on image
def plot_one_box(x, img, color=None, label=None, line_thickness=None, idx=0):
     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 # line thickness
     color = color or [random.randint(0, 255) for _ in range(3)]
     color = colors_box[idx]
     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
     cv2.rectangle(img, c1, c2, color, thickness=tl)
     if label:
        tf = max(tl - 1, 1) # font thickness
     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
     cv2.rectangle(img, c1, c2, color, -1) # filled
     cv2.putText(img, label, c1, 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
     return img


# decode one sequence to visual locations
def decode(coordinate_str, type='box'):

    # find numbers
    locations = np.array([int(i) for i in re.findall(r"\d+", coordinate_str)])

    if type == 'box':
        locations = locations.reshape(-1, 4)
    elif type == 'cocokeypoint':
        locations = locations.reshape(-1, 18, 2)
        visible = np.ones((locations.shape[0], 18, 1))
        eq_0_idx = np.where(locations[:, :, 0] * locations[:, :, 1] == 0)
        visible[eq_0_idx] = 0
        locations = np.concatenate([locations, visible], axis=-1)
        for i in range(locations.shape[0]):
            if locations[i, 2, -1] == 0 or locations[i, 5, -1] == 0:
                locations[i, 1, -1] = 0
    elif type == 'crowdpose':
        locations = locations.reshape(-1, 14, 2)
        visible = np.ones((locations.shape[0], 14, 1))
        eq_0_idx = np.where(locations[:, :, 0] * locations[:, :, 1] == 0)
        visible[eq_0_idx] = 0
        locations = np.concatenate([locations, visible], axis=-1)
    elif type == 'mask':
        locations = []
        for c_str in coordinate_str.split('m0'):
            c_str = ''.join(re.split(r'm\d+', c_str))
            mask_coord = np.array([int(i) for i in re.findall(r"\d+ ", c_str)])
            if len(mask_coord) != 0:
                locations.append(mask_coord.reshape(-1, 1, 2))
    else:
        raise NotImplementedError

    return locations


# process raw sequences inferred by VisorGPT
def to_coordinate(file_path, ctn=True):

    if isinstance(file_path, list):
        texts = [i.strip().replace(' ##', '') for i in file_path]
    else:
        with open(file_path, 'r') as file:
            texts = [i.strip().replace(' ##', '') for i in file.readlines()]

    location_list = []
    classname_list = []
    type_list = []
    valid_sequences = []
    cnt = 0
    print('to coordinate ...')

    for ste in tqdm(texts):
        cnt += 1
        if 'box' in ste:
            type = 'box'
        elif 'key point' in ste:
            type = 'cocokeypoint' if '; 18 ;' in ste else 'crowdpose'
        elif 'mask' in ste:
            type = 'mask'
        else:
            raise NotImplementedError

        if '[SEP]' not in ste:
            continue

        try:
            if ctn:
                temp = ste[:ste.index('[SEP]')].split(' ; ')[5].split('] ')
                classnames = []
                for t in temp:
                    classnames.append(t.split(' xmin ')[0].split(' m0')[0][2:])
                classnames = classnames[:-1]
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[5], type=type)

            else:
                classnames = ste[:ste.index('[SEP]')].split(' ; ')[5].split(' , ')
                locations = decode(ste[:ste.index('[SEP]')].split(' ; ')[6], type=type)
        except:
            pass
        else:
            valid_sequences.append(ste[:ste.index('[SEP]')])
            location_list.append(locations)
            classname_list.append(classnames)
            type_list.append(type)

    with open('valid_sequences.txt', 'w') as file:
        [file.write(i.split('[CLS] ')[-1] + '\n') for i in valid_sequences]

    return location_list, classname_list, type_list, valid_sequences

# visualize object locations on a canvas
def visualization(location_list, classname_list, type_list, save_dir='debug/', save_fig=False):

    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print('visualizing ...')
    for b, (loc, classnames, type) in tqdm(enumerate(zip(location_list, classname_list, type_list))):
        canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50

        if len(loc) != len(classnames):
            continue
        
        if type == 'box':
            for i in range(loc.shape[0]):
                canvas = plot_one_box(loc[i], canvas, label=classnames[i], idx=i)

        elif type == 'cocokeypoint':
            for i in range(loc.shape[0]):
                for j in range(loc.shape[1]):
                    x, y, v = loc[i, j]
                    if v != 0:
                        cv2.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)
                for j in range(17):
                    lim = limbSeq_coco[j]
                    cur_canvas = canvas.copy()

                    Y = [loc[i][lim[0] - 1][0], loc[i][lim[1] - 1][0]]
                    X = [loc[i][lim[0] - 1][1], loc[i][lim[1] - 1][1]]

                    if loc[i][lim[0] - 1][-1] == 0 or loc[i][lim[1] - 1][-1] == 0:
                        continue

                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[j])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        elif type == 'crowdpose':
            for i in range(loc.shape[0]):
                for j in range(loc.shape[1]):
                    x, y, _ = loc[i, j]
                    if x != 0 and y != 0:
                        cv2.circle(canvas, (int(x), int(y)), 4, colors[j], thickness=-1)
                for j in range(13):
                    lim = limbSeq_cp[j]
                    cur_canvas = canvas.copy()

                    Y = [loc[i][lim[0] - 1][0], loc[i][lim[1] - 1][0]]
                    X = [loc[i][lim[0] - 1][1], loc[i][lim[1] - 1][1]]

                    if (Y[0] == 0 and X[0] == 0) or (Y[1] == 0 and X[1] == 0):
                        continue

                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[j])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        elif type == 'mask':
            for i in range(len(loc)):
                color = [random.randint(0, 255) for _ in range(3)]
                xmin, ymin, xmax, ymax = loc[i][:, :, 0].min(), loc[i][:, :, 1].min(), loc[i][:, :, 0].max(), loc[i][:, :, 1].max()
                cur_canvas = canvas.copy()
                cv2.fillPoly(cur_canvas, [loc[i]], color)
                cur_canvas = plot_one_box((xmin, ymin, xmax, ymax), cur_canvas, color=color, label=classnames[i])
                canvas = cv2.addWeighted(canvas, 0.5, cur_canvas, 0.5, 0)
        else:
            raise NotImplementedError
        if save_fig:
            cv2.imwrite(f'{save_dir}/test_{b}.png', canvas[..., ::-1])
            
    return canvas[..., ::-1]

# to json output
def to_json(location_list, classname_list, type_list, valid_sequences):

    ret_json_box = {'bboxes': [], 'sequences': []}
    ret_json_mask = {'masks': [], 'sequences': []}
    ret_json_keypoint = {'keypoints': [], 'sequences': []}
    print('to json ...')
    for loc, classnames, type, seq in tqdm(zip(location_list, classname_list, type_list, valid_sequences)):
        ins_list = []
        kpt_list = []
        mask_list = []
        seq_list = []
        if len(loc) != len(classnames):# or len(classnames) > 8:
            continue

        if type == 'box':
            for i in range(loc.shape[0]):
                # xmin, ymin, xmax, ymax = loc[i]
                # area = (xmax - xmin) * (ymax - ymin)
                # compute area and omit very small one due to the synthesis ability of AIGC
                # if area < 32**2:
                #     continue

                dic = {classnames[i]: loc[i].tolist()}
                ins_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)

        elif type == 'cocokeypoint' or type == 'crowdpose':
            for i in range(loc.shape[0]):
                # compute validate key points and omit the less one, as the synthesis ability of AIGC
                # if loc[i, :, -1].sum() <= 4:
                #     continue

                # compute area and omit very small one due to the synthesis ability of AIGC
                # xmin, ymin, xmax, ymax = loc[i, :, 0].min(), loc[i, :, 1].min(), loc[i, :, 0].max(), loc[i, :, 1].max()
                # area = (xmax - xmin) * (ymax - ymin)
                # if area < 32 ** 2:
                #     continue

                dic = {classnames[i]: loc[i][:, :].tolist()}
                kpt_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)

        elif type == 'mask':
            for i in range(len(loc)):

                # xmin, ymin, xmax, ymax = loc[i][:, :, 0].min(), loc[i][:, :, 1].min(), loc[i][:, :, 0].max(), loc[i][:, :, 1].max()
                # area = (xmax - xmin) * (ymax - ymin)
                # if area < 32 ** 2:
                #     continue

                dic = {classnames[i]: loc[i].tolist()}
                mask_list.append(dic)
                if len(seq_list) == 0:
                    seq_list.append(seq)
        else:
            raise NotImplementedError

        if len(ins_list) != 0:
            ret_json_box['bboxes'].append(ins_list)
            ret_json_box['sequences'].append(seq_list)
        if len(kpt_list) != 0:
            ret_json_keypoint['keypoints'].append(kpt_list)
            ret_json_keypoint['sequences'].append(seq_list)
        if len(mask_list) != 0:
            ret_json_mask['masks'].append(mask_list)
            ret_json_mask['sequences'].append(seq_list)

    return [ret_json_box, ret_json_mask, ret_json_keypoint]


def gen_cond_mask(texts, ctn):
    location_list, classname_list, type_list, valid_sequences = to_coordinate(texts, ctn)
    ret_mask = visualization(location_list, classname_list, type_list, None, False)
    ret_json = to_json(location_list, classname_list, type_list, valid_sequences)
    return ret_mask, ret_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='debug')
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    location_list, classname_list, type_list, valid_sequences = to_coordinate(args.file_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # visualization
    if args.visualize:
        visualization(location_list, classname_list, type_list, args.save_dir)

    # to json data
    rets = to_json(location_list, classname_list, type_list, valid_sequences)

    for ret, flag in zip(rets, ['box', 'mask', 'keypoint']):
        save_path = args.file_path.split('/')[-1].split('.')[0] + f'_{flag}.json'
        with open('files/' + save_path, 'w') as file:
            json.dump(ret, file, indent=2)



