#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


import lib.data.fewshot
import lib.data.ovdshot
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT

from collections import defaultdict

import numpy as np

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
    
from sklearn.metrics import precision_recall_curve
from sklearn import metrics as sk_metrics

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import json

def load_and_remap_category_ids(annotation_path, output_path):
    """Loads annotation file, remaps category IDs if needed, and saves to a temp file."""
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Check if category IDs start from 0
    min_category_id = min([c["id"] for c in data["categories"]])
    if min_category_id == 0:
        #print(f"Category IDs in {annotation_path} start from 0. Remapping to start from 1.")

        # Create mapping {0 -> 1, 1 -> 2, ...}
        category_mapping = {c["id"]: c["id"] + 1 for c in data["categories"]}

        # Update category IDs
        for c in data["categories"]:
            c["id"] = category_mapping[c["id"]]
        
        # Update annotations
        for ann in data["annotations"]:
            ann["category_id"] = category_mapping[ann["category_id"]]

    # Save fixed JSON to a new file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    #print(f"Saved fixed annotations to {output_path}")
    return output_path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Moves up from ./tools/
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

#print('base_path',BASE_DIR)
#print('path',DATASET_DIR)
#print(os.path.join(DATASET_DIR, "dataset2/annotations/test.json"))

register_coco_instances("dataset2_test", {}, os.path.join(DATASET_DIR, "dataset2/annotations/test.json"), os.path.join(DATASET_DIR, "dataset2/test"))
register_coco_instances("dataset2_1shot", {}, os.path.join(DATASET_DIR, "dataset2/annotations/1_shot.json"), os.path.join(DATASET_DIR, "dataset2/train"))
register_coco_instances("dataset2_5shot", {}, os.path.join(DATASET_DIR, "dataset2/annotations/5_shot.json"), os.path.join(DATASET_DIR, "dataset2/train"))
register_coco_instances("dataset2_10shot", {}, os.path.join(DATASET_DIR, "dataset2/annotations/10_shot.json"), os.path.join(DATASET_DIR, "dataset2/train"))

register_coco_instances("dataset3_test", {}, os.path.join(DATASET_DIR, "dataset3/annotations/test.json"), os.path.join(DATASET_DIR, "dataset3/test"))
register_coco_instances("dataset3_1shot", {}, os.path.join(DATASET_DIR, "dataset3/annotations/1_shot.json"), os.path.join(DATASET_DIR, "dataset3/train"))
register_coco_instances("dataset3_5shot", {}, os.path.join(DATASET_DIR, "dataset3/annotations/5_shot.json"), os.path.join(DATASET_DIR, "dataset3/train"))
register_coco_instances("dataset3_10shot", {}, os.path.join(DATASET_DIR, "dataset3/annotations/10_shot.json"), os.path.join(DATASET_DIR, "dataset3/train"))

temp_dir = os.path.join(BASE_DIR, "temp_annotations") 
# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

for i in[1,5,10]:
    dataset_name = 'dataset1_%dshot' % i
    annotation_file = os.path.join(DATASET_DIR,'dataset1/annotations', f"{i}_shot.json")
    temp_annotation_file = os.path.join(temp_dir, f"{i}_shot_fixed.json")
    fixed_annotations = load_and_remap_category_ids(annotation_file, temp_annotation_file)
    register_coco_instances(dataset_name, {}, fixed_annotations, os.path.join(DATASET_DIR, "dataset1/train"))

temp_test_annotation_file = os.path.join(temp_dir, "test_fixed.json")
fixed_test_annotation_file = load_and_remap_category_ids(os.path.join(DATASET_DIR,"dataset1/annotations/test.json"), temp_test_annotation_file)
register_coco_instances('dataset1_test', {}, fixed_test_annotation_file, os.path.join(DATASET_DIR,"dataset1/test"))


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        if 'OpenSet' in cfg.MODEL.META_ARCHITECTURE:
            if 'lvis' in dataset_name:
                evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
            else:
                dtrain_name = cfg.DATASETS.TRAIN[0]
                # for coco14 FSOD benchmark
                if 'coco' in dataset_name:
                    seen_cnames = SEEN_CLS_DICT['fs_coco14_base_train']
                else:
                    seen_cnames = SEEN_CLS_DICT[dtrain_name]
                all_cnames = ALL_CLS_DICT[dtrain_name]
                unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
                evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
                                                seen_cnames=seen_cnames, unseen_cnames=unseen_cnames,
                                                all_cnames=all_cnames))
            return DatasetEvaluators(evaluator_list)
    

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DE.CONTROLLER = args.controller

    cfg.freeze()
    default_setup(cfg, args)
    print(cfg.DATASETS.TEST)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )