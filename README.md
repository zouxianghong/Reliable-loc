## PatchAugNet and SDP-Rerank

by Xianghong Zou

### Benchmark Datasets (refer to PointNetVLAD and Spectral GV)
* Oxford dataset (for baseline config training/testing: PointNetVLAD, PPT-Net, Minkloc3DV2)
* NUS (in-house) Datasets (for testing: PointNetVLAD, PPT-Net, Minkloc3DV2)
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)
* Self-Collected Datasets
  * wuhan hankou (for training/testing: PointNetVLAD, PPT-Net, Minkloc3DV2, PatchAugNet)
  * whu campus (for testing: PointNetVLAD, PPT-Net, Minkloc3DV2, PatchAugNet)

* MulRan Dataset (refer to Spectral GV)
  * Sejong01/Sejong02 (for training: EgoNN, LCDNet, LoGG3D-Net)
  * DCC01/DCC02 (for testing: EgoNN, LCDNet, LoGG3D-Net)
* KITTI 360 Dataset (refer to Spectral GV)
  * Seq 09 (for testing: EgoNN, LCDNet, LoGG3D-Net)
* Apollo SouthBay Dataset (refer to Spectral GV)
  * BaylandsToSeafood / ColumbiaPark / HighWay237 / MathildaAVE / SanJoseDowntown (for training: EgoNN, LCDNet, LoGG3D-Net)
  * Sunnyvale (for testing: EgoNN, LCDNet, LoGG3D-Net)

### Project Code
#### Pre-requisites
* Use docker image:
```
docker pull zouxh22135/pc_loc:v1
```
* Install Teaser++ in container:
```
apt install cmake libeigen3-dev libboost-all-dev
cd libs
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus
mkdir build
cd build
cmake ..
make
ctest
cmake -DTEASERPP_PYTHON_VERSION=3.6 ..
make teaserpp_python
cd python
pip install .
```
Note: 1. change the files in cmake/ once network errors happen;

#### Dataset set-up
* Download the zip file of the Oxford RobotCar and 3-Inhouse benchmark datasets found [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx) and extract the folder.
* Generate pickle files: We store the positive and negative point clouds to each anchor on pickle files that are used in our training and evaluation codes. The files only need to be generated once. The generation of these files may take a few minutes.
* Note: please check dataset info in 'datasets/dataset_info.py'
* Datasets defined in 'datasets/dataset_info.py', you can switch datasets by '--dataset' argument:
  * oxford
  * university, residential, business
  * hankou, campus
  * sejong, dcc
  * kitti360
  * excl_sunnyvale, sunnyvale
```
# For Oxford RobotCar / 3-Inhouse Datasets
python datasets/place_recognition_dataset.py

# For Self-Collected Dataset
python datasets/scene_dataset.py

# For MulRan-Sejong: training
python datasets/mulran/generate_training_tuples.py
python datasets/place_recognition_dataset.py

# For MulRan-DCC: testing
python datasets/mulran/generate_evaluation_sets.py
python datasets/place_recognition_dataset.py

# For KITTI-360
python datasets/kitti360/generate_evaluation_sets.py
python datasets/place_recognition_dataset.py
```

#### Place Recognition: Training and Evaluation
* Build the third parties
```
cd libs/pointops && python setup.py install && cd ../../
cd libs/chamfer_dist && python setup.py install && cd ../../
cd libs/emd_module && python setup.py install && cd ../../
cd libs/KNN_CUDA && python setup.py install && cd ../../
```

* Train / Eavaluate PointNetVLAD / PPT-Net / PatchAugNet
```
# Train PointNetVLAD / PPT-Net / Minkloc3D V2 / PatchAugNet on Oxford
python place_recognition/train_place_recognition.py --config configs/[pointnet_vlad / pptnet_origin / patch_aug_net].yaml --dataset oxford

# Evaluate PointNetVLAD / PPT-Net / Minkloc3D V2 / PatchAugNet on Oxford, and save top k
python place_recognition/evaluate.py --model_type [model type] --weight [weight pth file] --dataset oxford --exp_dir [exp_dir]

Note: model types include [pointnet_vlad / pptnet / pptnet_l2_norm / minkloc3d_v2 / patch_aug_net]
      datasets include [oxford / university / residential / business / hankou / campus]
```

#### Place Recognition: Training and Evaluation
* Run:
```
python monte_carlo_loc/main.py --dataset [dataset] --start_idx 0 --use_sgv --use_yaw --min_reliable_value 0.0005 --loc_mode -1
Note: datasets includ [cs_college / info_campus / zhongshan_park / jiefang_road / yanjiang_road1 / yanjiang_road2]
```

#### Citation
If you find the code or trained models useful, please consider citing:
```
@article{zou2025reliable,
  title={Reliable-loc: Robust sequential LiDAR global localization in large-scale street scenes based on verifiable cues},
  author={Zou, Xianghong and Li, Jianping and Wu, Weitong and Liang, Fuxun and Yang, Bisheng and Dong, Zhen},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={224},
  pages={287--301},
  year={2025},
  publisher={Elsevier}
}
```

#### Acknowledgement
Our code refers to [PointNetVLAD](), [PPT-Net](https://github.com/mikacuy/pointnetvlad), [Minkloc3DV2](), [EgoNN](), [LCDNet](), [LoGG3D-Net](), [Rank-PointRetrieval]() and [Spectral GV]().
