# Advanced Machine Learning 2022-2023 - Final Project

This repo was created for the final exam of the Advanced Machine Learning course of Politecnico di Torino. 

The group is composed of the students n°: **s307107**, **s307013**, **s307340**

The project aimed to understand an Anomaly Detection algorithm on industrial pieces called **Patchcore**[1]. This repo contain the code of the reimplementation of this algorithm based on the github of the reference paper[1] as a simplified version found in github[2].

## Part 1: Patchcore Implementation

The students where asked to reimplement patchcore in order to understand how patchcore works and find similar results as in the article. 

The notebook **MyPatchcore.ipynb** show the training and testing phases of the algorithm as the plotting of some statistics in order to compare the results throughout many hyperparameters. 

The algorithm is able to calculate the score of an anomaly and define the piece as good or not good. Moreover we implemented the segmentation part to identify where the anomaly is detected on the piece. 


## Part 2: Extension

For the extension, we tried to use the clip[3] pretrained network. Then we compared the results with the previous implementation. It can be find on ImprovedPatchcore

## References
[1] Roth et al., “Towards Total Recall in Industrial Anomaly Detection”, in CVPR 2022

[2] https://github.com/rvorias/ind_knn_ad/tree/05f46178545ddb9564bac8f872a83fccea8f4418

[3] https://openai.com/blog/clip/