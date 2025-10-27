# CASIAB Dataset for Simple-CCReID

## Overview
This directory contains the CASIA Gait Database Dataset B (CASIAB) formatted for use with the Simple-CCReID framework.

## Dataset Format
The dataset uses three text files:
- `train.txt`: Training set
- `query.txt`: Query set for evaluation
- `gallery.txt`: Gallery set for evaluation

Each line in these files has three columns separated by spaces:
```
<tracklet_path> <person_id> <angle_label>
```

Example:
```
track_seq/001/bg-01/000 001 000
track_seq/001/bg-01/018 001 018
track_seq/001/bg-01/036 001 036
```

Where:
- `tracklet_path`: Path to the directory containing video frames
- `person_id`: Person identity (e.g., 001, 002, ...)
- `angle_label`: Viewing angle in degrees (e.g., 000=0°, 018=18°, 036=36°, etc.)

## Angle Labels
The CASIAB dataset uses 11 different viewing angles:
- 000, 018, 036, 054, 072, 090, 108, 126, 144, 162, 180

These represent camera viewing angles from 0° to 180° in 18° increments.

## Usage with Simple-CCReID

### Training
To train a model on CASIAB:
```bash
python main.py --cfg configs/res50_cels_cal_16x4.yaml --dataset casiab --root <path_to_datasets>
```

### Key Differences from CCVID
While CCVID uses clothes labels to represent different clothing of the same person, CASIAB uses angle labels to represent different viewing angles of the same person. However, the implementation maintains full API compatibility:

| Aspect | CCVID | CASIAB |
|--------|-------|--------|
| Third column semantic | Clothes label | Angle label |
| Use case | Clothes-changing Re-ID | Gait recognition / View-invariant Re-ID |
| Internal variable | `clothes_id` | `angle_id` |
| Exposed attribute | `num_train_clothes` | `num_train_clothes` (mapped from angles) |
| Exposed attribute | `pid2clothes` | `pid2clothes` (mapped from angles) |

### Implementation Notes
- The CASIAB dataset loader (`data/datasets/casiab.py`) uses "angle" semantics internally
- For compatibility with the training framework, angle-related attributes are mapped to "clothes"-compatible names
- The `pid2clothes` matrix becomes `pid2angles` semantically, representing which angles belong to which person
- No changes to training code are required - the angle labels are treated as "view classes" similar to how clothes labels are treated as "clothes classes"

### Dataset Structure
```
Dataset/CASIAB/
├── README.md              # This file
├── train.txt             # Training set annotations
├── query.txt             # Query set annotations
├── gallery.txt           # Gallery set annotations
└── track_seq/            # Video frames (not included in repository)
    ├── 001/
    │   ├── bg-01/
    │   │   ├── 000/      # Person 001, condition bg-01, angle 0°
    │   │   ├── 018/      # Person 001, condition bg-01, angle 18°
    │   │   └── ...
    │   └── ...
    └── ...
```

## Statistics
- Training set: 5,728 tracklets
- Query set: 2,456 tracklets  
- Gallery set: 13,640 tracklets
- Total unique persons: 124
- Viewing angles: 11 (0° to 180° in 18° increments)

## Citation
If you use the CASIAB dataset, please cite the original work:
```
@inproceedings{yu2006framework,
  title={A framework for evaluating the effect of view angle, clothing and carrying condition on gait recognition},
  author={Yu, Shiqi and Tan, Daoliang and Tan, Tieniu},
  booktitle={18th International Conference on Pattern Recognition (ICPR'06)},
  pages={441--444},
  year={2006},
  organization={IEEE}
}
```
