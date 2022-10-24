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
* orange : resnet101
* blue : resnet50
* gray : resnet34
* blue : resnet18

|model|FLOPs(M)|Params(M)|Accuracy|
|:--:|:--:|:--:|:--:|
|resnet18|281.70|11.17|90.77%|
|resnet34|585.32|21.79|92.47%|
|resnet50|661.00|25.55|92.67%|
|resnet101|1267.58|44.54|93.30%|

# compression
## resnet34
|model|compression ratio|method|FLOPs(M)|Params(M)|Accuracy|
|:--:|:--:|:--:|:--:|:--:|:--:|
|resnet18|-|-|281.7M|11.17|90.77%|
|resnet34|-|-|585.32|21.79|92.47%|
|resnet34|0.5|L1|281.42|10.33|92.13%|
|resnet34|0.5|L2|280.86|10.23|91.72%|
|resnet34|0.5|GM|286.12|10.73|91.14%|

## resnet50
|model|compression ratio|method|FLOPs(M)|Params(M)|Accuracy|
|:--:|:--:|:--:|:--:|:--:|:--:|
|resnet18|-|-|281.7M|11.17|90.77%|
|resnet34|-|-|585.32|21.79|92.47%|
|resnet50|-|-|661.00|25.55|92.67%|
|resnet50|0.1|L1|571.66|22.17|93.15%|
|resnet50|0.1|L2|571.62|22.16|93.11%|
|resnet50|0.1|GM|573.53|22.45|93.11%|
|resnet50|0.5|L1|267.48|10.48|92.61%|
|resnet50|0.5|L2|269.02|10.47|92.70%|
|resnet50|0.5|GM|275.22|11.88|92.52%|

## resnet101
|model|compression ratio|method|FLOPs(M)|Params(M)|Accuracy|
|:--:|:--:|:--:|:--:|:--:|:--:|
|resnet18|-|-|281.7M|11.17|90.77%|
|resnet34|-|-|585.32|21.79|92.47%|
|resnet50|-|-|661.00|25.55|92.67%|
|resnet101|-|-|1267.58|44.54|93.30%|
|resnet101|0.7|L1|257.70|9.20|92.30%|
|resnet101|0.7|L2|259.70|9.24|92.47%|
|resnet101|0.7|GM|264.30|10.76|89.92%|
|resnet101|0.45|L1|553.16|19.49|92.92%|
|resnet101|0.45|L2|553.70|19.54|93.16%|
|resnet101|0.45|GM|564.56|21.12|-|
|resnet101|0.4|L1|621.86|21.86|93.26%|
|resnet101|0.4|L2|621.78|21.89|93.20%|
|resnet101|0.4|GM|633.40|23.39|93.06%|
