# NTIRE 2025 CD-FSOD Challenge @ CVPR Workshop

We are the **team** of the **NTIRE 2025 Cross-Domain Few-Shot Object Detection (CD-FSOD) Challenge** at the **CVPR Workshop**.

- 🏆 **Track**: `closed-source track`
- 🎖️ **Award**: **3rd Place**

🔗 [NTIRE 2025 Official Website](https://cvlai.net/ntire/2025/)  
🔗 [NTIRE 2025 Challenge Website](https://codalab.lisn.upsaclay.fr/competitions/21851)  
🔗 [CD-FSOD Challenge Repository](https://github.com/lovelyqian/NTIRE2025_CDFSOD)

![CD-FSOD Task](https://upload-images.jianshu.io/upload_images/9933353-3d7be0d924bd4270.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## Overview

This repository contains our solution for the `closed-source track` of the NTIRE 2025 CD-FSOD Challenge.  

We strictly use the same training data as our baseline method (COCO)—without introducing any extra pretraining models or additional source domain data.

---

## The Environments
The evaluation environments we adopted are recorded in the following section. Below are the system requirements and setup instructions for reproducing the evaluation environment.

### Required Environment Setup
We suggest using Anaconda for environment management. Here's how to set up the environment for the challenge:

- **Step 1**: conda environment create:
  ```bash
    conda create -n cdfsod python=3.9
    conda activate cdfsod
- **Step 2**: install other libs:
  ```bash
    cd NTIRE2025_CDFSOD
    pip install -r requirements.txt
    pip install -e ./
or take it as a reference based on your original environments.

## The Validation Datasets
We take COCO as source data and ArTaxOr, Clipart1k, and DeepFish as validation datasets.

The target datasets could be easily downloaded in the following links: 
- [Dataset and Weights Link from Google Drive](https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q?usp=drive_link)

## The Test Datasets

**The testing datasets could be easily downloaded in the following links:**
- **[Dataset Link from Google Drive](https://drive.google.com/drive/folders/1Pewv7HYacwD5Rrknp5EiAdw8vMbaaFAA?usp=sharing)**

After downloading all the necessary validation datasets, make sure they are organized as follows:

```bash
|NTIRE2025_CDFSOD/datasets/
|--clipart1k/
|   |--annotations
|   |--test
|   |--train
|--ArTaxOr/
|   |--annotations
|   |--test
|   |--train
|--......
```
And the weights should be organized as follows:
```bash
|NTIRE2025_CDFSOD/weights/
|--trained/
|   |--vitl_0089999.pth
|--background/
|   |--background_prototypes.vitl14.pth
```

## Build Prototypes for Testing datasets
As the environment is ready, generate class prototypes for the testing datasets
```
sbatch run_pkl.slurm
```
## Run the Baseline Model
```
sbatch run_main.slurm
```
### Ensure Each Dataset Has the Correct Category Mapping
```
python convert_id -x 1.py
python convert_id -x 5.py
python convert_id -x 10.py
```

## Citation
If you use our method or codes in your research, please cite:
```
@inproceedings{fu2025ntire, 
  title={NTIRE 2025 challenge on cross-domain few-shot object detection: methods and results,
  author={Fu, Yuqian and Qiu, Xingyu and Ren, Bin and Fu, Yanwei and Timofte, Radu and Sebe, Nicu and Yang, Ming-Hsuan and Van Gool, Luc and others},
  booktitle={CVPRW},
  year={2025}
}
```