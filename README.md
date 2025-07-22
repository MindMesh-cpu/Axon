# Axon
EEG Signal Classification with Deep Temporal-Channel Fusion Networks
This project implements a deep learning model for EEG signal classification using advanced temporal and channel attention mechanisms. It includes preprocessing with bandpass filtering, data augmentation techniques like mixup and channel dropout, and a multi-scale convolutional architecture designed to capture rich temporal and spatial features in EEG data. The code supports training, validation, and testing with best model checkpointing.

基于深度时序与通道融合的脑电信号分类
本项目实现了一个基于深度学习的脑电（EEG）信号分类模型，融合了多尺度时序卷积和通道注意力机制。包含带通滤波预处理、数据增强（mixup、通道丢弃等），以及专为提取脑电时空特征设计的卷积网络结构。支持训练、验证和测试，自动保存最佳模型。
验证集准确率在 60%–70% 之间，其他测试集的最佳结果在 50%–60% 左右。由于实验仅在阿里云提供的 NVIDIA T4 小规模 GPU 上进行，尚未进行充分的超参数调优，和充分的消融实验，初步结果略微超过基线EEGNET。
虽然模型结构创新性有限（主要为模块化拼接和在时序提取上混合多层卷积提高感受野），但在效率与准确率之间实现了较好平衡。在此基础上，结合迁移学习与不同精神状态下 MI 信号的进一步划分与研究，该方法在 MI 识别的实际应用中具有较大的潜力与可拓展性。

大致结构：
Temporal Feature Extraction（时间特征提取）-> Channel Feature Extraction（通道特征提取）-> Attention Modules（注意力机制）-> Fusion Module（特征融合）->Classifier（分类器）

