import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat
from tools.utils import mkdir_if_missing, write_json, read_json


class CASIAB(object):
    """ CASIAB

    CASIA Gait Database Dataset B
    Note: This dataset uses angle labels instead of clothes labels.
    The angle represents the viewing angle (e.g., 018 means 18 degrees).
    """
    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root, 'CASIAB')
        self.train_path = osp.join(self.root, 'train.txt')
        self.query_path = osp.join(self.root, 'query.txt')
        self.gallery_path = osp.join(self.root, 'gallery.txt')
        self._check_before_run()
 
        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_angles, pid2angles, _ = \
            self._process_data(self.train_path, relabel=True)
        angles2label = self._angles2label_test(self.query_path, self.gallery_path)
        query, num_query_tracklets, num_query_pids, num_query_imgs, num_query_angles, _, _ = \
            self._process_data(self.query_path, relabel=False, angles2label=angles2label)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, num_gallery_angles, _, _ = \
            self._process_data(self.gallery_path, relabel=False, angles2label=angles2label)

        # slice each full-length video in the trainingset into more video clip
        train_dense = self._densesampling_for_trainingset(train, sampling_step)
        # In the test stage, each video sample is divided into a series of equilong video clips with a pre-defined stride.
        recombined_query, query_vid2clip_index = self._recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = self._recombination_for_testset(gallery, seq_len=seq_len, stride=stride)
       
        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs 
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_angles = num_train_angles + len(angles2label)
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets 

        logger = logging.getLogger('reid.dataset')
        logger.info("=> CASIAB loaded")
        logger.info("Dataset statistics:")
        logger.info("  ---------------------------------------------")
        logger.info("  subset       | # ids | # tracklets | # angles")
        logger.info("  ---------------------------------------------")
        logger.info("  train        | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_angles))
        logger.info("  train_dense  | {:5d} | {:11d} | {:9d}".format(num_train_pids, len(train_dense), num_train_angles))
        logger.info("  query        | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_angles))
        logger.info("  gallery      | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_angles))
        logger.info("  ---------------------------------------------")
        logger.info("  total        | {:5d} | {:11d} | {:9d}".format(num_total_pids, num_total_tracklets, num_total_angles))
        logger.info("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        logger.info("  ---------------------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = query
        self.gallery = gallery

        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        # For compatibility with training code that expects num_train_clothes
        self.num_train_clothes = num_train_angles
        # For compatibility with training code that expects pid2clothes
        self.pid2clothes = pid2angles

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_path):
            raise RuntimeError("'{}' is not available".format(self.train_path))
        if not osp.exists(self.query_path):
            raise RuntimeError("'{}' is not available".format(self.query_path))
        if not osp.exists(self.gallery_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_path))

    def _angles2label_test(self, query_path, gallery_path):
        pid_container = set()
        angles_container = set()
        with open(query_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, angle_label = new_line.split()
                angle = '{}_{}'.format(pid, angle_label)
                pid_container.add(pid)
                angles_container.add(angle)
        with open(gallery_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, angle_label = new_line.split()
                angle = '{}_{}'.format(pid, angle_label)
                pid_container.add(pid)
                angles_container.add(angle)
        pid_container = sorted(pid_container)
        angles_container = sorted(angles_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        angles2label = {angle:label for label, angle in enumerate(angles_container)}

        return angles2label

    def _process_data(self, data_path, relabel=False, angles2label=None):
        tracklet_path_list = []
        pid_container = set()
        angles_container = set()
        with open(data_path, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                tracklet_path, pid, angle_label = new_line.split()
                tracklet_path_list.append((tracklet_path, pid, angle_label))
                angle = '{}_{}'.format(pid, angle_label)
                pid_container.add(pid)
                angles_container.add(angle)
        pid_container = sorted(pid_container)
        angles_container = sorted(angles_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        if angles2label is None:
            angles2label = {angle:label for label, angle in enumerate(angles_container)}

        num_tracklets = len(tracklet_path_list)
        num_pids = len(pid_container)
        num_angles = len(angles_container)

        tracklets = []
        num_imgs_per_tracklet = []
        pid2angles = np.zeros((num_pids, len(angles2label)))

        for tracklet_path, pid, angle_label in tracklet_path_list:
            img_paths = glob.glob(osp.join(self.root, tracklet_path, '*')) 
            img_paths.sort()

            angle = '{}_{}'.format(pid, angle_label)
            angle_id = angles2label[angle]
            pid2angles[pid2label[pid], angle_id] = 1
            if relabel:
                pid = pid2label[pid]
            else:
                pid = int(pid)
            
            # Extract camera id from path
            # For CASIAB, we'll use a simple scheme based on tracklet path
            # This can be adjusted based on actual CASIAB structure
            camid = 0  # Default camera id, adjust if needed
            
            num_imgs_per_tracklet.append(len(img_paths))
            # Note: using angle_id as the 4th element (clothes_id position for compatibility)
            tracklets.append((img_paths, pid, camid, angle_id))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_angles, pid2angles, angles2label

    def _densesampling_for_trainingset(self, dataset, sampling_step=64):
        ''' Split all videos in training set into lots of clips for dense sampling.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, angle_id)
            sampling_step (int): sampling step for dense sampling

        Returns:
            new_dataset (list): output dataset
        '''
        new_dataset = []
        for (img_paths, pid, camid, angle_id) in dataset:
            if sampling_step != 0:
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    new_dataset.append((img_paths, pid, camid, angle_id))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            new_dataset.append((img_paths[idx*sampling_step:], pid, camid, angle_id))
                        else:
                            new_dataset.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid, angle_id))
            else:
                new_dataset.append((img_paths, pid, camid, angle_id))

        return new_dataset

    def _recombination_for_testset(self, dataset, seq_len=16, stride=4):
        ''' Split all videos in test set into lots of equilong clips.

        Args:
            dataset (list): input dataset, each video is organized as (img_paths, pid, camid, angle_id)
            seq_len (int): sequence length of each output clip
            stride (int): temporal sampling stride

        Returns:
            new_dataset (list): output dataset with lots of equilong clips
            vid2clip_index (list): a list contains the start and end clip index of each original video
        '''
        new_dataset = []
        vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
        for idx, (img_paths, pid, camid, angle_id) in enumerate(dataset):
            # start index
            vid2clip_index[idx, 0] = len(new_dataset)
            # process the sequence that can be divisible by seq_len*stride
            for i in range(len(img_paths)//(seq_len*stride)):
                for j in range(stride):
                    begin_idx = i * (seq_len * stride) + j
                    end_idx = (i + 1) * (seq_len * stride)
                    clip_paths = img_paths[begin_idx : end_idx : stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, angle_id))
            # process the remaining sequence that can't be divisible by seq_len*stride        
            if len(img_paths)%(seq_len*stride) != 0:
                # reducing stride
                new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
                for i in range(new_stride):
                    begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                    end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                    clip_paths = img_paths[begin_idx : end_idx : new_stride]
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, angle_id))
                # process the remaining sequence that can't be divisible by seq_len
                if len(img_paths) % seq_len != 0:
                    clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                    # loop padding
                    while len(clip_paths) < seq_len:
                        for index in clip_paths:
                            if len(clip_paths) >= seq_len:
                                break
                            clip_paths.append(index)
                    assert(len(clip_paths) == seq_len)
                    new_dataset.append((clip_paths, pid, camid, angle_id))
            # end index
            vid2clip_index[idx, 1] = len(new_dataset)
            assert((vid2clip_index[idx, 1]-vid2clip_index[idx, 0]) == math.ceil(len(img_paths)/seq_len))

        return new_dataset, vid2clip_index.tolist()
