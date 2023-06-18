## Awesome-BEV-Perception

本仓库由[公众号【自动驾驶之心】](https://mp.weixin.qq.com/s?__biz=Mzg2NzUxNTU1OA==&mid=2247542481&idx=1&sn=c6d8609491a128233c3c3b91d68d22a6&chksm=ceb80b18f9cf820e789efd75947633aec9d2f1e8b58c29e5051c05a64b21ae63c244d54886a1&token=11182364&lang=zh_CN#rd) 团队整理，欢迎关注，一览最前沿的技术分享！

自动驾驶与AI学习社区：欢迎加入国内首个自动驾驶开发者社区！这里有最全面有效的自动驾驶与AI学习路线（感知/定位/融合）和自动驾驶与AI公司内推机会！

## 一、**综述** | Overview

### 1.**基于BEV的3D目标检测综述** | A review of BEV-based 3D target detection

Vision-Centric BEV Perception: A Survey

### 2.**2022.9.12 BEV感知最新综述** | BEV Perception Update Roundup

Delving into the Devils of Bird’s-eye-view Perception: A Review, Evaluation and Recipe

[[Code]](https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe)

### 3.**2023 面向BEV检测的视觉-雷达融合综述**

### A review of vision-radar fusion for BEV detection

Vision-RADAR fusion for Robotics BEV Detections: A Survey

### 4.**2023** **自动驾驶环绕视图的3D目标检测综述**

### A review of 3D target detection for self-driving surround view

Surround-View Vision-based 3D Detection for Autonomous Driving: A Survey

## 二、**基于相机的BEV** | Camera-based BEV

### 1.**基于相机的BEV感知方法列表** | List of camera-based BEV sensing methods

Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D 

[project](https://github.com/nv-tlabs/lift-splat-shoot)

BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View 

 [project](https://github.com/HuangJunJie2017/BEVDet) 

BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection

 [project](https://github.com/HuangJunJie2017/BEVDet) 

BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection

[project](https://github.com/Megvii-BaseDetection/BEVDepth)

DSGN: Deep Stereo Geometry Network for 3D Object Detection 

 [project](https://github.com/dvlab-research/DSGN) 

LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-Based 3D Detector 

[project](https://github.com/xy-guo/LIGA-Stereo)

Is Pseudo-Lidar Needed for Monocular 3D Object Detection? 

[project](https://github.com/TRI-ML/dd3d) 

Inverse perspective mapping simplifies optical flow computation and obstacle detection 

Deep Learning based Vehicle Position and Orientation Estimation via Inverse Perspective Mapping Image 

Learning to Map Vehicles into Bird’s Eye View 

Monocular 3D Vehicle Detection Using Uncalibrated Traffic Cameras through Homography 

Driving Among Flatmobiles: Bird-Eye-View Occupancy Grids From a Monocular Camera for Holistic Trajectory Planning 

Understanding Bird’s-Eye View of Road Semantics Using an Onboard Camera

 [project](https://github.com/ybarancan/BEV_feat_stitch)

Automatic dense visual semantic mapping from street-level imagery

Stacked Homography Transformations for Multi-View Pedestrian Detection 

Cross-View Semantic Segmentation for Sensing Surroundings 

[project](https://github.com/pbw-Berwin/View-Parsing-Network)

FISHING Net: Future Inference of Semantic Heatmaps In Grids 

NEAT: Neural Attention Fields for End-to-End Autonomous Driving

[project](https://github.com/autonomousvision/neat)

Projecting Your View Attentively: Monocular Road Scene Layout Estimation via Cross-View Transformation 

[project](https://github.com/JonDoe-297/cross-view)

Bird’s-Eye-View Panoptic Segmentation Using Monocular Frontal View Images 

[project](https://github.com/robot-learning-freiburg/PanopticBEV)

BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers 

 [project](https://github.com/zhiqi-li/BEVFormer)

PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark 

[project](https://github.com/OpenPerceptionX/PersFormer_3DLane)

PETR: Position Embedding Transformation for Multi-View 3D Object Detection

[project](https://github.com/megvii-research/PETR)

DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries 

[project](https://github.com/WangYueFt/detr3d)

Translating Images into Maps 

[project](https://github.com/avishkarsaha/translating-images-into-maps)

GitNet: Geometric Prior-based Transformation for Birds-Eye-View Segmentation 

PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images

[project](https://github.com/megvii-research/PETR)

ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection supplemental 

[project](https://github.com/saic-vul/imvoxelnet)

MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones 

 [project](https://github.com/Tai-Wang/Depth-from-Motion)

FIERY: Future Instance Prediction in Bird's-Eye View From Surround Monocular Cameras 

BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving 

[project](https://github.com/zhangyp15/BEVerse)

### 2.**基于环绕单目图像的BEV外观和Occupancy信息估计**

### Estimation of BEV Appearance and Occupancy Information Based on Surrounding Monocular Images

Estimation  of  Appearance  and  Occupancy  Information  in  Bird’s  EyeView  from  Surround  Monocular  Images

[[Code]](https://uditsinghparihar.github.io/APP_OCC/)

###3.**去相机参数的BEV表示方法** | BEV representation of de-camera parameters

Multi-Camera Calibration Free BEV Representation for 3D Object Detection

### 4.**BEVFormerV2**

BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision

### 5.**无预标定的相机进行多视图subject registration**

From a Bird’s Eye View to See: Joint Camera and Subject Registration without the Camera Calibration

## 三、**基于LiDAR的BEV** | LiDAR-based BEV

### 1.**基于LiDAR的BEV感知方法列表** | List of LiDAR-based BEV sensing methods

VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

SECOND: Sparsely Embedded Convolutional Detection 

[project](https://github.com/traveller59/second.pytorch)

Center-Based 3D Object Detection and Tracking 

[project](https://github.com/tianweiy/CenterPoint)

PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection

[project](https://github.com/sshaoshuai/PV-RCNN) 

PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection

[project](https://github.com/open-mmlab/OpenPCDet)

Structure Aware Single-Stage 3D Object Detection From Point Cloud

[project](https://github.com/skyhehe123/SA-SSD)

Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection

[project](https://github.com/djiajunustc/Voxel-R-CNN)

Object DGCNN: 3D Object Detection using Dynamic Graphs

Voxel Transformer for 3D Object Detection 

Embracing Single Stride 3D Object Detector With Sparse Transformer / paper / supplemental 

[project](https://github.com/tusen-ai/SST)

AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds 

PointPillars: Fast Encoders for Object Detection From Point Clouds 

### 2.**BEV-MAE：基于点云的预训练框架** | Point cloud-based pre-training framework

BEV-MAE: Bird's Eye View Masked Autoencoders for Outdoor Point Cloud Pre-training

### 3.**BEV-SAN：**利用切片注意力进行精确的BEV 3D目标检测

### BEV-SAN: Accurate BEV 3D target detection using slicing attention

BEV-SAN: Accurate BEV 3D Object Detection via Slice Attention Networks

### 4.**2D目标检测和LiDAR联合训练（2.5D点）**

###  2D target detection and LiDAR joint training (2.5D points)

Objects as Spatio-Temporal 2.5D points

### 5.**BEV-LGKD：统一框架**针对BEV 3D目标检测任务应用知识蒸馏

BEV-LGKD: A Unified LiDAR-Guided Knowledge Distillation Framework for BEV 3D Object Detection

## 四、**BEV融合** | BEV Fusion

### 1.**BEV融合方法列表** | List of BEV fusion methods

Unifying Voxel-based Representation with Transformer for 3D Object Detection

[project](https://github.com/dvlab-research/UVTR)

MVFuseNet: Improving End-to-End Object Detection and Motion Forecasting Through Multi-View Fusion of LiDAR Data

UniFormer: Unified Multi-view Fusion Transformer for Spatial-Temporal Representation in Bird's-Eye-View 

BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

[project](https://github.com/mit-han-lab/bevfusion)

BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework

[project](https://github.com/ADLab-AutoDrive/BEVFusion)

### 2.**X-Align相机和LiDAR的BEV特征融合改进**

### BEV feature fusion improvement with X-Align camera and LiDAR

X-Align Cross-Modal Cross-View Alignment for Bird’s-Eye-View Segmentation

### 3.**Radar和LiDAR BEV融合系统** | Radar and LiDAR BEV fusion system

RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection System

### 4.**BEV下的多模态融合方法汇总** | Summary of multimodal fusion methods under BEV

PointPainting: Sequential Fusion for 3D Object Detection (CVPR'19)

[project](https://github.com/AmrElsersy/PointPainting)

3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection (ECCV'20) 

[project](https://github.com/rasd3/3D-CVF)

FUTR3D: A Unified Sensor Fusion Framework for 3D Detection (Arxiv'22)

[project](https://github.com/Tsinghua-MARS-Lab/futr3d)

MVP: Multimodal Virtual Point 3D Detection (NIPS'21)

[project](https://tianweiy.github.io/mvp/)

PointAugmenting: Cross-Modal Augmentation for 3D Object Detection

[project](https://github.com/VISION-SJTU/PointAugmenting)

FusionPainting: Multimodal Fusion with Adaptive Attention for 3D Object Detection 

[project](https://github.com/Shaoqing26/FusionPainting)

Unifying Voxel-based Representation with Transformer for 3D Object Detection 

[project](https://github.com/dvlab-research/UVTR)

TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers 

[project](https://github.com/XuyangBai/TransFusion)

AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection 

[project](https://github.com/zehuichen123/AutoAlignV2)

AutoAlignV2: Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection 

[project](https://github.com/zehuichen123/AutoAlignV2)

CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection 

[project](https://github.com/mrnabati/CenterFusion)

MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection

[project](https://github.com/SxJyJay/MSMDFusion)

## 五、**BEV下的多任务学习方法汇总**

### 1.Summary of multi-task learning methods under BEV

FIERY: Future Instance Prediction in Bird’s-Eye View from Surround Monocular Cameras 

[project](https://github.com/wayveai/fiery)

StretchBEV: Stretching Future Instance Prediction Spatially and Temporally 

[project](https://github.com/kaanakan/stretchbev)

BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving 

[project](https://github.com/zhangyp15/BEVerse)

M^2^BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation

[project](https://nvlabs.github.io/M2BEV/)

STSU: Structured Bird’s-Eye-View Traffic Scene Understanding from Onboard Images 

[project](https://github.com/ybarancan/STSU)

BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers

[project](https://github.com/zhiqi-li/BEVFormer)

Ego3RT: Learning Ego 3D Representation as Ray Tracing

[project](https://github.com/fudan-zvg/Ego3RT)

PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images 

[project](https://github.com/megvii-research/PETR)

PolarFormer: Multi-camera 3D Object Detection with Polar Transformers

[project](https://github.com/fudan-zvg/PolarFormer)

## 六、**PV2BEV方法汇总** | PV2BEV method summary

### 1.**基于depth方法的PV2BEV汇总** | Summary of PV2BEV based on depth method

OFT: Orthographic Feature Transform for Monocular 3D Object Detection 

[project](https://github.com/tom-roddick/oft)

CaDDN: Categorical Depth Distribution Network for Monocular 3D Object Detection 

[project](https://github.com/TRAILab/CaDDN)

DSGN: Deep Stereo Geometry Network for 3D Object Detection

[project](https://github.com/dvlab-research/DSGN)

Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D

[project](https://nv-tlabs.github.io/lift-splat-shoot/)

PanopticSeg: Bird’s-Eye-View Panoptic Segmentation Using Monocular Frontal View Images 

[project](http://panoptic-bev.cs.uni-freiburg.de/)

FIERY: Future Instance Prediction in Bird’s-Eye View from Surround Monocular Cameras 

[project](https://github.com/wayveai/fiery)

LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-based 3D Detector

[project](https://xy-guo.github.io/liga/)

ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection

[project](https://github.com/saic-vul/imvoxelnet)

BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View 

[project](https://github.com/HuangJunJie2017/BEVDet)

M^2^BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Bird’s-Eye View Representation

[project](https://nvlabs.github.io/M2BEV/)

StretchBEV: Stretching Future Instance Prediction Spatially and Temporally

[project](https://github.com/kaanakan/stretchbev)

DfM: Monocular 3D Object Detection with Depth from Motion

[project](https://github.com/Tai-Wang/Depth-from-Motion)

BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection 

[project](https://github.com/HuangJunJie2017/BEVDet)

BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving

[project](https://github.com/zhangyp15/BEVerse)

MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones

[project](https://github.com/Tai-Wang/Depth-from-Motion)

Putting People in their Place: Monocular Regression of 3D People in Depth 

[Code](https://github.com/Arthur151/ROMP) [Project](https://arthur151.github.io/BEV/BEV.html) [Video](https://youtu.be/Q62fj_6AxRI) [RH Dataset](https://github.com/Arthur151/Relative_Human)

### 2.**基于霍夫变换的PV2BEV方法汇总**

### Summary of PV2BEV methods based on Hough transform

IPM: Inverse perspective mapping simplifies optical flow computation and obstacle detection 

DSM: Automatic Dense Visual Semantic Mapping from Street-Level Imagery 

MapV: Learning to map vehicles into bird’s eye view

BridgeGAN: Generative Adversarial Frontal View to Bird View Synthesis 

[project](https://github.com/xinge008/BridgeGAN)

VPOE: Deep learning based vehicle position and orientation estimation via inverse perspective mapping image

3D-LaneNet: End-to-End 3D Multiple Lane Detection

The Right (Angled) Perspective: Improving the Understanding of Road Scenes Using Boosted Inverse Perspective Mapping

Cam2BEV: A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird’s Eye View 

[project](https://github.com/ika-rwth-aachen/Cam2BEV)

MonoLayout: Amodal Scene Layout from a Single Image 

[project](https://github.com/hbutsuak95/monolayout)

MVNet: Multiview Detection with Feature Perspective Transformation 

[project](https://github.com/hou-yz/MVDet)

OGMs: Driving among Flatmobiles: Bird-Eye-View occupancy grids from a monocular camera for holistic trajectory planning 

[project](https://github.com/4DVLab/Vision-Centric-BEV-Perception/blob/main)

TrafCam3D: Monocular 3D Vehicle Detection Using Uncalibrated Traffic Camerasthrough Homography

[project](https://github.com/minghanz/trafcam_3d)

SHOT:Stacked Homography Transformations for Multi-View Pedestrian Detection 

HomoLoss: Homography Loss for Monocular 3D Object Detection

