import glob
import re

import os.path as osp
import os

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class STOAT(BaseImageDataset):
    """
    New Zealand (Waiheke Island and South Island) Stoat Dataset
    
    Dataset statistics:
    # train - South Island, gallery - Waiheke Island, query - Waiheke Island
    # identities: 56 (train) + 5 (gallery) + 5 (query)
    # images: 183 (train) + 13 (gallery) + 13 (query)
    """
    dataset_dir = "Stoat"

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(STOAT, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.aid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel = True)
        query = self._process_dir(self.query_dir, relabel = False)
        gallery = self._process_dir(self.gallery_dir, relabel = False)
        

        if verbose:
            print("=> Stoat loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        

    def _process_dir(self, dir_path, relabel=False):
        set_name = dir_path.split("/")[-1]
        
        
        # Construct a list of image paths.
        image_paths_list = []
        for file in sorted(os.listdir(dir_path)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dir_path, file)
                image_paths_list.append(image_path)
        
        # special_cameras = {"CREK" : 1000, "FC01" : 1001, "FC11" : 1002, "GC34" : 1003, "P164" : 1004}
        # pattern = re.compile(r'\d+_[0-9a-zA-Z]+_\d+')
        
        label = 0
        aid2label = dict()
        aid_container = set()
        camid_container = set()
        dataset = []
        
        for img_path in image_paths_list:
            img_name, file_extension = os.path.splitext(os.path.basename(img_path))
            components_list = img_name.split("_")
            aid, camid = int(components_list[0]), components_list[1]
            aid_container.add(aid)
            camid_container.add(camid)
            if aid not in aid2label:
                aid2label[aid] = label
                label += 1
            
            if relabel:
                aid = aid2label[aid]
                
            camid = -1
            dataset.append((img_path, self.aid_begin + aid, camid, 0))

        return dataset