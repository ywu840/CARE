import logging
import os
import sys
import os.path as osp
from config import cfg

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        seed = str(cfg.SOLVER.SEED)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, f"train_log_{seed}.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, f"test_log_{seed}.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger