EEG Signal Classification Based on Deep Temporal–Channel Fusion Networks

My primary objective is to explore EEG signal classification for motor imagery (MI) tasks under small-sample conditions. After experimenting with several non-Transformer models, I decided to integrate strengths from multiple studies to construct a more interpretable network structure: temporal features are captured through multi-receptive-field convolutions applied on single-channel time series; cross-channel relationships are modeled via fully connected layers; and both temporal and spatial attention mechanisms are incorporated to enhance discriminative power.

In my experiments, I found that baseline models such as EEGNet often fail to converge or reproduce the reported results without extensive preprocessing. In contrast, my proposed structure, combined with bandpass filtering, MixUp, channel dropout, and other preprocessing and data augmentation techniques, can effectively fit the data and outperform the baselines. This demonstrates not only the gap between traditional models and practical applications, but also the feasibility of combining multi-receptive-field convolutions with attention mechanisms.

On the validation set, my model achieves an accuracy of 60%–70%, while the best test results are around 55%–65%, showing a noticeable improvement over EEGNet. The experiments were conducted on a small-scale NVIDIA T4 GPU provided by Alibaba Cloud, and extensive hyperparameter tuning or systematic ablation studies have not yet been performed.

Although the model structure is not highly innovative (mainly relying on modular stacking and multi-scale convolutions to strengthen temporal feature extraction), it achieves a good balance between efficiency and accuracy. In the next step, I plan to incorporate an improved lightweight attention mechanism on top of EEGPT embeddings to enhance cross-subject generalization and further investigate MI signal classification under different mental states.

Overall structure:
Temporal feature extraction → Channel feature modeling → Attention modules → Fusion module → Classifier
