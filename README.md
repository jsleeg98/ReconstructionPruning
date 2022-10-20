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

## KITTI compression
|model|FLOPs(G)|FLOPs ratio|parameters|mAP|
|:--:|:--:|:--:|:--:|:--:|
|original|16.414|100%|61,626,499|0.7704|
|compress 0.3|7.184|43.77%|12,149,889|0.8116|
|compress 0.5|4.776|29.10%|7,182,739|0.7820|
|compress 0.7|3.228|19.67%|5,869,315|0.7367|
|compress 0.9|0.913|5.57%|2,490,926|0.4848|

---

# model
* resnet18
* resnet34
* resnet50
* resnet101

# train result
![acc_graph](/README_image/resnet_original_acc.svg)

|model|FLOPs(M)|Params(M)|Accuracy|
|:--:|:--:|:--:|:--:|
|resnet18|281.70|11.17|90.77%|
|resnet34|585.32|21.79|92.47%|
|resnet50|661.00|25.55|92.67%|
|resnet101|1267.58|44.54|93.30%|

