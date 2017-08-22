# PointCNN

Created by <a href="http://yangyan.li" target="_blank">Yangyan Li</a>, Rui Bu, <a href="http://www.mcsun.cn" target="_blank">Mingchao Sun</a>, and <a href="http://www.cs.sdu.edu.cn/~baoquan/" target="_blank">Baoquan Chen</a> from Shandong University.

### Introduction

We present a simple and general framework for feature learning from point cloud. The key to the success of CNNs is the convolution operator that is capable of leveraging spatially-local correlation in data represented densely in grids (e.g. images). However, point cloud are irregular and unordered, thus a direct convolving of kernels against the features associated with the points will result in deserting the shape information while being variant to the orders. To address these problems, we propose to learn a X-transformation from the input points, and then use it to simultaneously weight the input features associated with the points and permute them into latent potentially canonical order, before the element-wise product and sum operations are applied. The proposed method is a generalization of typical CNNs into learning features from point cloud, thus we call it *PointCNN*. Experiments show that PointCNN achieves on par or better performance than state-of-the-art methods on multiple challenging benchmark datasets and tasks.

See our <a href="http://arxiv.org/abs/1801.07791" target="_blank">research paper on arXiv</a> for more details.

### Code Organization
The core X-Conv and PointCNN architecture are defined in ./pointcnn.py.

The network/training/data augmentation hyperparameters for classification tasks are defined in ./pointcnn_cls/\*.py, for segmentation tasks are defined in ./pointcnn_cls/\*.py

### Usage

Commands for training and testing ModelNet40 classification:
```
cd data_conversions
python3 ./download_datasets.py -d modelnet
cd ../pointcnn_cls
./train_val_modelnet.sh -g 0 -x modelnet_x2_l4
```

Commands for training and testing ShapeNet Parts segmentation:
```
cd data_conversions
python3 ./download_datasets.py -d shapenet_partseg
cd ../pointcnn_seg
./train_val_shapenet.sh -g 0 -x shapenet_x8_2048_fps
./test_shapenet.sh -g 0 -x shapenet_x8_2048_fps -l ../../models/seg/pointcnn_seg_shapenet_x8_2048_fps_xxxx/ckpts/iter-xxxxx -r 10
cd ..
python3 ./evaluate_seg.py -g ../data/shapenet_partseg/test_label -p ../data/shapenet_partseg/test_data_pred_10
```

Other datasets can be processed in a similar way.
