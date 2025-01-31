# Occluded Contrastive Learning for Self-supervised Pre-training


This repository is a PyTorch implementation of the *One Leaf Reveals the Season: Occlusion-Based Contrastive Learning with Semantic-Aware Views for Efficient Visual Representation* (submitted)

![workflow](./imgs/workflow.png)

This paper proposes a scalable and straightforward pre-training paradigm for efficient visual conceptual representation called masked image contrastive learning (MiCL). Our MiCL approach is simple: we randomly mask patches to generate different views within an image and contrast them among a mini-batch of images. The core idea behind MiCL consists of two designs. First, masked tokens have the potential to significantly diminish the conceptual redundancy inherent in images, and create distinct views with substantial fine-grained differences on the semantic concept level instead of the instance level. Second, contrastive learning is adept at extracting high-level semantic conceptual features during the pre-training, circumventing the high-frequency interference and additional costs associated with image reconstruction. Importantly, MiCL learns highly semantic conceptual representations efficiently without relying on hand-crafted data augmentations or additional auxiliary modules. Empirically, MiCL demonstrates high scalability with Vision Transformers, as the ViT-L/16 can complete pre-training in 133 hours using only 4 A100 GPUs, achieving 85.8% accuracy in downstream fine-tuning tasks.

The code in this repo is copied/modified from [MAE](https://github.com/facebookresearch/mae).

## Pre-training

The pre-training instruction is in [PRETRAIN.md](./PRETRAIN.md).

