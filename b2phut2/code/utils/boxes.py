import numpy as np
import json


def extract_labels_boxes(meta):
    '''
    Extract the labels and boxes from the raw metadata.

    Parameters
    ----------
    meta : dict
        The metadata is what is contained in the dict from the pickle files
        provided with the project in e.g. data/SVHN/train/labels.pkl.

    Returns
    -------
    labels : list
        Contains the integers of the digits present in the image.
    boxes : list
        Contains the tuples (x1, x2, y1, y2) of coordinates of bounding boxes
        associated to each digit in labels.

    '''

    n = len(meta['label'])  # Number of digits in image

    labels = []  # Digits present in image
    boxes = []  # bboxes present in image

    # Extract digit boxes and labels
    for jj in range(n):
        labels.append(int(meta['label'][jj]))
        y1 = meta['top'][jj]
        y2 = y1 + meta['height'][jj]
        x1 = meta['left'][jj]
        x2 = x1 + meta['width'][jj]

        boxes.append((x1, x2, y1, y2))

    boxes = np.asarray(boxes)

    return labels, boxes


def extract_outer_box(sample, padding=0.3):
    '''
    Extract outer box from individuals boxes.

    Parameters
    ----------
    sample : Dict
        Output of the dataloader.
    padding : float
        Percentage of padding around the bounding boxe containg
        all digits. Should be in range [0, 1].

    Returns
    -------
    outer_bbox : Tuple
        Tuple (x1, x2, y1, y2) of coordinates of bounding boxes
        associated to the digits sequence.

    '''
    img_shape = np.asarray(sample['image']).shape
    boxes = sample['metadata']['boxes']

    x1_tot = np.min(boxes[:, 0])
    x2_tot = np.max(boxes[:, 1])
    y1_tot = np.min(boxes[:, 2])
    y2_tot = np.max(boxes[:, 3])

    x1_tot -= padding / 2 * (x2_tot - x1_tot)
    x2_tot += padding / 2 * (x2_tot - x1_tot)
    y1_tot -= padding / 2 * (y2_tot - y1_tot)
    y2_tot += padding / 2 * (y2_tot - y1_tot)

    x1_tot = max(0, x1_tot)
    x2_tot = min(x2_tot, img_shape[1] - 1)
    y1_tot = max(0, y1_tot)
    y2_tot = min(y2_tot, img_shape[0] - 1)

    outer_bbox = (x1_tot, x2_tot, y1_tot, y2_tot)

    return outer_bbox


def read_bbox(sample, padding=0.3):
    '''
    read the bbox.json file
    '''
    id = sample['metadata']['img_id']
    # TODO: change this hardcoded string
    # TODO: pull from master and change this to train vs val
    data_dir = '../../data/r_50_7200/coco_humanware_v1_1553272293_val/bbox.json'

    with open(data_dir, 'r') as f:
        val_dict = json.load(f)

    boxes = val_dict[id]['bbox']
    print("bbox from json boxes {} {}".format(type(boxes), boxes))

    img_shape = np.asarray(sample['image']).shape

    # coco dataset is `x_top_left`, `y_top_left`, `width` and `height`

    x1_tot = boxes[0]
    x2_tot = boxes[1]
    y1_tot = boxes[2]
    y2_tot = boxes[3]

    x1_tot -= padding / 2 * (x2_tot - x1_tot)
    x2_tot += padding / 2 * (x2_tot - x1_tot)
    y1_tot -= padding / 2 * (y2_tot - y1_tot)
    y2_tot += padding / 2 * (y2_tot - y1_tot)

    x1_tot = max(0, x1_tot)
    x2_tot = min(x2_tot, img_shape[1] - 1)
    y1_tot = max(0, y1_tot)
    y2_tot = min(y2_tot, img_shape[0] - 1)

    outer_bbox = (x1_tot, x2_tot, y1_tot, y2_tot)
    print("***********************")
    print("outer_bbox", outer_bbox)
    print("***********************")

    # get the original bbox from json file in FAST RCNN output
    # print(val_dict[id]['bbox'])
    return outer_bbox
