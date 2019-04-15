#!/usr/bin/env python3
"""
Script to convert bbox.json to metadata.pkl file
"""

import argparse
import json
import pickle


def convert(instance_file, bbox_file, output_file, original_metadata=None):
    with open(instance_file) as fob:
        instance_file_data = json.load(fob)
    with open(bbox_file) as fob:
        bbox_file_data = json.load(fob)
    if original_metadata is not None:
        with open(original_metadata, 'rb') as fob:
            original_metadata = pickle.load(fob)

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
        last_score = new_bbox_format[image_id]['score']
        if last_score < score:
            new_bbox_format[image_id] = box

    metadata = {}
    # loop on instances and create metadata
    for instance in instance_file_data['images']:
        image_id = instance['id']
        bbox_info = new_bbox_format.get(image_id, None)
        if bbox_info is None:
            # use a fake bbox
            bbox_info = {
                'bbox': [372.32, 374.93, 61.69, 62.60]}
        bbox_info = bbox_info['bbox']
        label = [1]
        if original_metadata is not None:
            label = original_metadata[image_id]['metdata']['label']
        label_len = len(label)
        metadata[image_id] = {
            'filename': instance['file_name'],
            'metadata': {
                'label': label,
                'left': [bbox_info[0]]*label_len,
                'top': [bbox_info[1]]*label_len,
                'height': [bbox_info[2]]*label_len,
                'width': [bbox_info[3]]*label_len,
            }
        }
    with open(output_file, 'wb') as fob:
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
    parser.add_argument(
        "--original-metadata",
        type=str,
        default=None,
        required=False,
        help="Supply when testing end-to-end",
    )
    args = parser.parse_args()
    convert(args.instance_file, args.bbox_file,
            args.output_file, args.original_metadata)
