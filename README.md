# Project
Reconstruct structured pruning

# model
* YOLOv3

# train result
## KITTI dataset
* iou_thres : 0.5
* nms_thres : 0.45
* conf_thres : 0.2
* weight_decay : 0.0001
* epoch : 200
* lr : 0.0001 -> 0.00001(100 epoch) -> 0.000001(150 epoch)
* Best MAP : 0.7704

|-|fold1|fold2|fold3|fold4|fold5|
|:--:|:--:|:--:|:--:|:--:|:--:|
|MAP|0.7432|0.7608|0.7494|0.7588|0.7704|

|fold5 test MAP graph|fold5 train loss graph|
|:--:|:--:|
|![test_graph](/KITTI/train_result/original_5/graph/yolov3_test_MAP_graph_fold5.png)|![train_loss](/KITTI/train_result/original_5/graph/yolov3_train_loss_graph_fold5.png)|
