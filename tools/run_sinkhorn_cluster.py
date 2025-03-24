import torch
from pprint import pprint
import numpy as np
import os.path as osp
import fire
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from detectron2.data import MetadataCatalog, DatasetCatalog
from collections import defaultdict, Counter

import lib.data.fewshot
import lib.data.ovdshot
import lib.data.lvis
from lib.prototype_learner import PrototypeLearner

from detectron2.data.datasets import register_coco_instances
import json
import os
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


"""
dataset = {
    'labels': [],
    'class_tokens': [],
    'avg_patch_tokens': [], 
    'image_id': [],
    'boxes': [],
    'areas': [],
    'skip': 0
}
"""

def accuracy_score(match):
    return (match.sum() / len(match)).item()


def main(inp, 
        num_prototypes=10,
        token_type='pat', # pat, cls
        momentum=0.002, 
        epochs=30,
        batch_size=512,
        queue_size=8192,
        normalize='yes',
        device=0,
        save='yes',
        save_tokens='no',
        oneshot_sample_pool=30
        ):
    oneshot = 'novel_oneshot_' in inp
    kwargs = locals()
    pprint(kwargs)
    if device != 'cpu': device = int(device)
    print('load and preprocess dataset')
    dataset = torch.load(inp)
    dname = osp.basename(inp).split('.')[0]
    if dname == "clipart1k_TEST":
        dname = "clipart1k_test"

    if 'label_names' in dataset:
        thing_classes = dataset['label_names']
    else:
        DatasetCatalog.get(dname)  # just to register some metas
        meta = MetadataCatalog.get(dname)
        thing_classes = meta.thing_classes

    is_background_tokens = False

    if len(dataset['avg_patch_tokens']) > 0:
        labels = torch.as_tensor(dataset['labels'])
        if len(dataset['avg_patch_tokens'][0].shape) == 1:
            tokens = torch.stack(dataset['avg_patch_tokens'])
        else:
            tokens = torch.cat(dataset['avg_patch_tokens'])
        assert len(tokens) == len(labels)
    else:
        # background tokens
        # in this case, thing_class will be invalid, ignore the class name output
        is_background_tokens = True

        def align_labels_from_patch_tokens(labels, patch_tokens):
            new_labels = []
            for l, p in zip(labels, patch_tokens):
                new_labels.extend([l, ] * len(p))
            return torch.as_tensor(new_labels)
        
        labels = align_labels_from_patch_tokens(dataset['labels'], dataset['patch_tokens'])
        labels -= 1
        tokens = torch.cat(dataset['patch_tokens'])
    
    if oneshot: 
        # these extracted samples are only for eval during one-shot training
        # true one-shot eval is conducted with ref images from BHRL
        tokens_count = defaultdict(list)
        origin_thing_classes = []
        
        for ci, cls_name in enumerate(thing_classes):
            ori_cls_name = cls_name.split('-')[0]
            if ori_cls_name not in origin_thing_classes:
                origin_thing_classes.append(ori_cls_name)
            tokens_count[ori_cls_name].append(((labels == ci).sum().item(), ci))
        
        for c, v in tokens_count.items():
            assert len(v) >= oneshot_sample_pool, f"{c} does not have enough instances!"
        
        tokens_count = {ori_cls_name: sorted(counts, reverse=True, key=lambda v: v[0])[:oneshot_sample_pool] 
                        for ori_cls_name, counts in tokens_count.items()}

        new_tokens, new_labels = [], []
        thing_classes = []
        class_count = 0

        for ori_cls_name in origin_thing_classes:
            for s, ci in tokens_count[ori_cls_name]:
                new_tokens.append(tokens[labels == ci])
                assert len(new_tokens[-1]) == s
                new_labels.append(torch.full([s,], class_count))
                thing_classes.append(f'{ori_cls_name}.{class_count}')
                class_count += 1

        tokens, labels = torch.cat(new_tokens), torch.cat(new_labels)

    full_tokens, full_labels = tokens, labels
    num_classes = len(torch.unique(labels))
    assert num_classes == labels.max().item() + 1
    print(f'total tokens {len(tokens)}, num classes {num_classes}')

    prototypes = PrototypeLearner(num_classes, num_prototypes, tokens.shape[-1], queue_size,
                                momentum=momentum, normalize=normalize == 'yes')
    prototypes.to(device)

    tdataset = torch.cat([tokens, labels[:, None]], dim=1)
    dloader = DataLoader(tdataset, batch_size=batch_size, shuffle=True)
    with tqdm(total=epochs * len(dloader), desc='learning prototypes') as bar:
        for _ in range(epochs):
            for batch in dloader:
                batch = batch.to(device)
                batch_tokens = batch[:, :-1]
                batch_labels = batch[:, -1]
                prototypes.update(batch_tokens, batch_labels)
                bar.update()

    print('self testing prototypes')
    prototypes = prototypes.cpu()
    pred_labels = torch.cat([prototypes.predict(x).cpu() for x in tqdm(torch.split(full_tokens, 3000))])
    # full_labels
    match = pred_labels == full_labels
        
    print(' --> Class Acc ')
    cls_accs = []
    for ci in torch.unique(labels).tolist():
        result = match[full_labels == ci]
        cls_accs.append(accuracy_score(result))
        print(f'- {thing_classes[ci]} = {cls_accs[-1]:.04f}')   

    print(f'cls acc = {np.mean(cls_accs):.04f}')
    kwargs['cls_acc'] = np.mean(cls_accs)
    ovr_acc = accuracy_score(match)
    print(f'   Overall = {ovr_acc:.04f}')
    kwargs['ovr_acc'] = ovr_acc
    pname = f'p{num_prototypes}'
    if save == 'yes':
        if is_background_tokens:
            dct = prototypes.values.cpu()
        else:
            dct = {'prototypes': prototypes.values.cpu().view(num_classes, num_prototypes, -1), 'kwargs': kwargs}

            # for one-shot, `label_names` will have duplicate values, and will need to be
            # dedup when loading
            dct['label_names'] = [thing_classes[ci] for ci in range(num_classes)]
            print(f'`label_names` are set to {dct["label_names"]}')

            if oneshot:
                dct['origin_label_names'] = origin_thing_classes

            if save_tokens == 'yes':
                dct['tokens'] = full_tokens
                dct['labels'] = full_labels 

        out_p = osp.splitext(inp)[0] + f'.p{num_prototypes}.sk.pkl'
        print(f'Saving to {out_p}')
        torch.save(dct, out_p)

if __name__ == "__main__":
    fire.Fire(main)