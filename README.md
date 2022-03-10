## Problem Statement

"How to improve the accuracy of Cross Domain Crowd Counting (CDCC) while addressing domain shift by using an improved model architecture that SFCN architecture."

## Project Objectives

1. Develop a new Convolutional Neural Network based crowd counting model to have a higher accuracy in CDCC.

2. Develop a complete application system that processes images and videos to estimate and output density maps and crowd counts.

3. Focus on automating the crowd density monitoring process with the use of an inexpensive hardware setup.

## Proposed CNN based Model Architecture

![Model Image](/Images/model.jpeg?raw=true "Proposed Model Architecture with Frontend and Backend Networks")

Proposed multi layered model architecture consists of two components as,

1) Frontend Network containing VGG 16 classifier and five feature generation levels as P5, P4, P3, P2, P1 for predicting density maps in high quality with multi-scale feature representation.

2) Backend network containing five multi-module branches and a dilationconvolution network for scale variation feature extractions to increase the accuracy in crowd count estimations while maintaining the resolution and high quality of generated density maps.
