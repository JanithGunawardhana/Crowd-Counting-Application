# Project Description

Automated crowd density monitoring is an emerging area of research. It is a vital technology that assists during recent disease outbreaks in preserving social distancing, crowd management and other widespread applications in public security and traffic control. Modern methods to count people in crowded scenes mainly rely on Convolutional Neural Network (CNN) based models. But the model’s ability to adapt for different domains which is referred to as cross domain crowd countingis a challenging task. To remedy this difficulty, many researchers used Spatial Fully Convolutional Network (SFCN) based crowd counting models with synthetic crowd scene data. They covered many image domains with few-shot learning to reduce the domain adaptation gap between source and target image domains. In this paper, we propose a new multi-layered model architecture instead of SFCN single-layered model architecture. The proposed model extracts more meaningful features in image scenes along with large scale variations to increase the accuracy in cross domain crowd counting. Furthermore, with extensive experiments using four real-world datasets and analysis, we show that the proposed multi-layered architecture performs well with synthetic image data and few-shot learning in reducing domain shifts.

## Problem Statement

"How to improve the accuracy of Cross Domain Crowd Counting (CDCC) while addressing domain shift by using an improved model architecture that SFCN architecture."

## Project Objectives

1. Develop a new Convolutional Neural Network based crowd counting model to have a higher accuracy in CDCC.

2. Develop a complete application system that processes images and videos to estimate and output density maps and crowd counts.

3. Focus on automating the crowd density monitoring process with the use of an inexpensive hardware setup.

## Proposed CNN based Model Architecture

![Model Image](https://github.com/JanithGunawardhana/Crowd-Counting-Application/blob/main/Images/model.jpeg "Proposed Model Architecture with Frontend and Backend Networks")

Proposed multi layered model architecture consists of two components as,

1) Frontend Network containing VGG 16 classifier and five feature generation levels as P5, P4, P3, P2, P1 for predicting density maps in high quality with multi-scale feature representation.

2) Backend network containing five multi-module branches and a dilationconvolution network for scale variation feature extractions to increase the accuracy in crowd count estimations while maintaining the resolution and high quality of generated density maps.
