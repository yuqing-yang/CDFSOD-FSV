import os
import json
import cv2
import random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

dataset_dir = "./datasets/dataset1/test"  
#pred_json_path = "output/vitl/dataset2_10shot/inference/coco_instances_results.json"
pred_json_path = "./dataset1_1shot.json"
ann_json_path = "./datasets/dataset1/annotations/test.json"
save_dir = "./vis_predictions"
os.makedirs(save_dir, exist_ok=True)

# === Load COCO objects ===
coco_gt = COCO(ann_json_path)
coco_pred = coco_gt.loadRes(pred_json_path)

img_ids = coco_gt.getImgIds()
random.seed(42)
sampled_img_ids = random.sample(img_ids, 10)

# === Visualize and Save ===
for img_id in sampled_img_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    file_path = os.path.join(dataset_dir, img_info['file_name'])
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw predictions
    anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))
    for ann in anns:
        x, y, w, h = ann["bbox"]
        score = ann.get("score", 1.0)
        cat_id = ann["category_id"]
        label = coco_gt.loadCats(cat_id)[0]["name"]

        if score <0.5:
            continue

        # Draw box + label
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save
    save_path = os.path.join(save_dir, f"pred_{img_info['file_name']}")
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")