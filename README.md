Lidar classification using a hierarchical deep neural network with iterative features
=========

Introduction
---------------
The purpose of the project is for semantic labeling airborne LiDAR point cloud.The project is built on [open3d]( http://www.open3d.org/) and [pytorch_geometric]( https://github.com/rusty1s/pytorch_geometric ). Open3d lib provides functions to handle point cloud with .pcd format. Pytorch_geometric provides the functions for deep learning on raw points.

Installation
---------------
 * Install Pytorch from [https://pytorch.org/]( https://pytorch.org/ )
 * Install Pytorch_geometric from [https://github.com/rusty1s/pytorch_geometric]( https://github.com/rusty1s/pytorch_geometric )
 * Install Open3D from [http://www.open3d.org/]( http://www.open3d.org/ )


Usage
---------------
### 1、Dataset
Lidar point cloud can be download from [ISPRS]( https://www2.isprs.org/commissions/comm3/wg4/3d-semantic-labeling.html ).
### 2、Preprocess data set
The LiDAR point cloud data set first converts to .pcd, and features and labels of the points are written to .feaures and .label ASCII files ( see samples.features and samples.labels ).
### 3、Paramets
Modify ./data/samples.json to set parameters, e.g. max epoch, batch size, learning rate, etc.
### 4、Train the model
Run
python train.py
### 5、Predict
Run
python predict.py
### 6、Interpolate the predict results to origin point cloud
Run
python interploate.py 
