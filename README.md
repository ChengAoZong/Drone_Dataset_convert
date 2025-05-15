# Drone_Dataset_convert
Converting UAV datasets of different formats into object detection format.

## 1、Convert DUT-Anti-UAV
prepare dataset as follow format:
```
[data_root]
├── test
│   ├── img
│   └── xml
├── train
│   ├── img
│   └── xml
└── val
    ├── img
    └── xml
```
```
python convert_DUT_Anti_UAV.py --data_root [data_root] --save_vis(optional)
```
the yolo format annotations will be save at [data_root]/train [data_root]/val and [data_root]/test