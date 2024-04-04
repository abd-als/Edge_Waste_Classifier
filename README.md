# Edge Waste Classifier

## Overview

This project focuses on real-time waste classification using Convolutional Neural Networks (CNNs) deployed on a Raspberry Pi 4B. The goal is to classify waste objects efficiently and accurately, making it suitable for edge computing applications.

## Dataset

The project utilizes the Recycled dataset created by Portland State University. This dataset provides a diverse collection of waste objects suitable for training and testing various machine learning models. Initially, the models are trained and evaluated using this dataset to establish a baseline performance.

The Roboflow Recyclable Items dataset supplements the training data, offering additional variety and depth to enhance model performance further.

- [Portland State University Recycled Dataset](http://web.cecs.pdx.edu/~singh/rcyc-web/dataset.html)
- [Roboflow Recyclable Items Dataset](https://universe.roboflow.com/recycle/recyclable-items/dataset/3)

## Approach

The notebook compares various machine learning models, including CNN architectures, for waste classification. Initially, these models are trained and evaluated using the Portland State University Recycled Dataset. The best-performing model, which was determined to be a CNN, is further refined and compared with other CNN architectures using the Roboflow dataset.

Once the best CNN architecture is identified, it is deployed on the Raspberry Pi for real-time waste classification.
