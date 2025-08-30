EEG Signal Classification with Deep Temporal-Channel Fusion Networks

This project implements a deep learning model for EEG signal classification, incorporating advanced temporal and channel attention mechanisms. The model includes preprocessing with bandpass filtering, data augmentation techniques (such as mixup and channel dropout), and a multi-scale convolutional architecture designed to capture rich temporal and spatial features in EEG data. It supports training, validation, and testing, with automatic checkpointing for the best model.

The validation accuracy ranges between 60%–70%, and the best test set results are around 55%–65%. The experiments were conducted using a small-scale NVIDIA T4 GPU provided by Alibaba Cloud, and further hyperparameter tuning and ablation studies have yet to be performed. The preliminary results slightly outperform the baseline EEGNet.

While the model structure is not highly innovative (mainly modular stacking and enhancing receptive fields with multi-layer convolutions for temporal extraction), it achieves a good balance between efficiency and accuracy. Building on this, the approach has great potential and scalability for practical applications in Motor Imagery (MI) signal recognition, especially when combined with transfer learning and further investigation into MI signal classification under different mental states.
