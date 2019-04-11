#!/usr/bin/env python3
"""
Script to convert bbox.json to metadata.pkl file
"""

import argparse
import json
import pickle


def convert(instance_file, bbox_file, output_file):
    with open(instance_file):
        instance_file_data = json.load(instance_file)
    with open(bbox_file):
        bbox_file_data = json.load(bbox_file)

    new_bbox_format = {}
    # loop and select the box with most score
    for box in bbox_file_data:
        image_id = box['image_id']
        category_id = box['category_id']

        # only interested in category 1
        if category_id != 1:
            continue

        score = box['score']
        if image_id not in new_bbox_format:
            new_bbox_format[image_id] = box
            continue
        last_score = new_bbox_format[image_id]['box']
        if last_score < score:
            new_bbox_format[image_id] = box

    metadata = {}
    # loop on instances and create metadata
    for instance in instance_file_data['images']:
        image_id = instance['id']
        bbox_info = new_bbox_format.get(image_id, None)
        if bbox_info is None:
            # use a fake bbox
            bbox_info = [372.32, 374.93, 61.69, 62.60]

        metadata[image_id] = {
            'filename': instance['file_name'],
            'metadata': {
                'label': [1],
                'left': [bbox_info[0]],
                'top': [bbox_info[1]],
                'height': [bbox_file[2]],
                'width': [bbox_file[3]],
            }
        }
    with open(output_file, 'rb') as fob:
        pickle.dump(metadata, fob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert bbox to metadata")
    parser.add_argument(
        "--bbox-file",
        type=str,
        help="bbox.json file path",
        required=True,
    )
    parser.add_argument(
        "--instance-file",
        type=str,
        help="Instances file path",
        required=True,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path",
    )
    args = parser.parse_args()
    convert(args.instance_file, args.bbox_file, args.output_file)
