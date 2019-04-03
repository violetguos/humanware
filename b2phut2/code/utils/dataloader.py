import os

from PIL import Image
from torch.utils import data

import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, SubsetRandomSampler

from utils.transforms import FirstCrop, Rescale, RandomCrop, ToTensor
from utils.misc import load_obj
from utils.boxes import extract_labels_boxes


class SVHNDataset(data.Dataset):

    def __init__(self, metadata, data_dir, transform=None, normalize_transform=None):
        '''
        Initialize the Dataset.

        Parameters
        ----------
        metadata : dict
        Each key in metadata will contain a filename and all associated
        metadata. The filename is with respect to the directory it's in,
        and metadata contains four 5 fields:
        * `label` - which lists all digits present in image, in order
        * `height`,`width`,`top`,`left` which list the corresponding pixel
        information about the digits bounding boxes.

        {0: {'filename': '1.png',
        'metadata': {'height': [219, 219],
        'label': [1, 9],
        'left': [246, 323],
        'top': [77, 81],
        'width': [81, 96]}},
        1: {'filename': '2.png',
        'metadata': {'height': [32, 32],
        'label': [2, 3],
        'left': [77, 98],
        'top': [29, 25],
        'width': [23, 26]}}, ...

        data_dir : str
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied on a sample.

        '''
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform
        self.normalize_transform = normalize_transform

    def __len__(self):
        '''
        Evaluate the length of the dataset object

        Returns
        -------
        int
            The length of the dataset.

        '''
        return len(self.metadata)

    def __getitem__(self, index):
        '''
        Get an indexed item from the dataset and return it.

        Parameters
        ----------
        index : int
            The index of the dataset

        Returns
        -------
        sample: dict
            sample['image'] contains the image array.
            The type may be a torch.tensor or ndarray depending on transforms
            sample['metadata'] will contain the metadata associated to the
            image. It can be one of ['labels','boxes','filename']


        '''
        'Generates one sample of data'

        img_name = os.path.join(self.data_dir,
                                self.metadata[index]['filename'])

        # Load data and get raw metadata (labels & boxes)
        image = Image.open(img_name)
        # FIXME: check why this image_1 is zero

        metadata_raw = self.metadata[index]['metadata']
        # inner boxes no longer present

        labels, boxes = extract_labels_boxes(metadata_raw)

        # need to add index
        # begin my hack
        metadata = {'labels': labels, 'boxes': boxes, 'filename': img_name, 'img_id': index}
        # end my hack

        # old B2 T2 implementation, keep records
        # metadata = {'labels': labels, 'boxes': boxes, 'filename': img_name}

        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)
            if self.normalize_transform:
                sample["image"] = self.normalize_transform(sample["image"])

        return sample


def prepare_dataloaders(dataset_split,
                        dataset_path,
                        metadata_filename,
                        batch_size=32,
                        sample_size=-1,
                        valid_split=0.1,
                        test_split=0.1,
                        num_worker=0,
                        extra_metadata_filename=None,
                        extra_dataset_dir=None):
    '''
    Utility function to prepare dataloaders for training.

    Parameters
    ----------
    dataset_split : str
        Any of 'train', 'extra', 'test'.
    dataset_path : str
        Absolute path to the dataset. (i.e. .../data/SVHN/train')
    metadata_filename : str
        Absolute path to the metadata pickle file.
    batch_size : int
        Mini-batch size.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.
    valid_split : float
        Returns a validation split of %size; valid_split*100,
        valid_split should be in range [0,1].

    Returns
    -------
    if dataset_split in ['train', 'extra']:
        train_loader: torch.utils.DataLoader
            Dataloader containing training data.
        valid_loader: torch.utils.DataLoader
            Dataloader containing validation data.

    if dataset_split in ['test']:
        test_loader: torch.utils.DataLoader
            Dataloader containing test data.

    '''

    assert dataset_split in ['train', 'test', 'extra'], "check dataset_split"

    metadata = load_obj(metadata_filename)

    #  dataset_path = datadir / dataset_split

    firstcrop = FirstCrop(0.3)
    downscale = Rescale((64, 64))
    random_crop = RandomCrop((54, 54))
    to_tensor = ToTensor()
    normalize = None
    # normalize = Normalize((0.434, 0.442, 0.473), (0.2, 0.202, 0.198))

    # Declare transformations

    transform = transforms.Compose([firstcrop,
                                    downscale,
                                    random_crop,
                                    to_tensor
                                    ])
    print("data_dir", dataset_path)
    dataset = SVHNDataset(metadata,
                          data_dir=dataset_path,
                          transform=transform, normalize_transform=normalize)

    # The extra file was given in block 2, we don't need it for block3 
    
    # if extra_metadata_filename is not None:
    #     extra_metadata = load_obj(extra_metadata_filename)
    #     extra_dataset = SVHNDataset(extra_metadata,
    #                                 data_dir=extra_dataset_dir,
    #                                 transform=transform, normalize_transform=normalize)
    #     dataset = ConcatDataset([dataset, extra_dataset])
    dataset_length = len(metadata)

    indices = np.arange(dataset_length)
    # Only use a sample amount of data
    if sample_size != -1:
        indices = indices[:sample_size]

    if dataset_split in ['train', 'extra']:

        # [train_subset, valid_subset, test_from_train_subset] = random_split(dataset,
        #                                                                     [,
        #                                                                      ,
        #                                                                      ])
        train_length = round(dataset_length * (1 - valid_split - test_split))
        valid_length = round(dataset_length * valid_split)

        train_idx = indices[:train_length]
        valid_idx = indices[train_length:train_length + valid_length]
        test_idx = indices[train_length + valid_length:]

        train_subset = Subset(dataset, train_idx)
        valid_subset = Subset(dataset, valid_idx)
        test_from_train_subset = Subset(dataset, test_idx)

        # Prepare a train and validation dataloader
        if extra_metadata_filename is not None:
            extra_metadata = load_obj(extra_metadata_filename)
            extra_dataset = SVHNDataset(extra_metadata,
                                        data_dir=extra_dataset_dir,
                                        transform=transform, normalize_transform=normalize)
            train_subset = ConcatDataset([train_subset, extra_dataset])
        train_loader = DataLoader(train_subset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_worker)
        valid_loader = DataLoader(valid_subset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_worker)
        test_from_train_loader = DataLoader(test_from_train_subset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_worker)

        return train_loader, valid_loader, test_from_train_loader

    elif dataset_split in ['test']:

        test_sampler = torch.utils.data.SequentialSampler(indices)

        # Prepare a test dataloader
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 shuffle=False,
                                 sampler=test_sampler)

        return test_loader
