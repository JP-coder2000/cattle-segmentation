# Analysis of Cow Behavior in Resting Areas Using Aerial Images

## Project Overview

This project aims to analyze the behavior of dairy cows in their resting areas through aerial images. We utilize data mining to find patterns and significant insights regarding the use of sand beds, such as how much time cows spend resting, how often they prefer certain spaces, and the postures they adopt in the beds.

## Project Objectives

### Business Objectives
- **Resting space efficiency**: Improve the usage of available beds to optimize the cows' rest time.
- **Facilitate decision-making**: Provide insights that allow the team to make more informed decisions regarding the management of resting spaces.

### Key Questions:
1. How many cows are resting in the beds?
2. How long do they stay in the beds?
3. Which beds are used more frequently than others?
4. How are the cows distributed when resting (standing or lying down)?

### Data Mining Objectives
- Classify aerial images to identify if a bed contains a standing cow, a lying cow, or if it is empty.
- Obtain metrics such as the average bed usage time per cow, the percentage of usage per bed, and the cows' resting status.

## Methodology

This project follows the CRISP-DM methodology (Cross-Industry Standard Process for Data Mining), covering the following phases:

1. **Business Understanding**: Define business objectives and key questions.
2. **Data Understanding**: Collect and analyze the images to understand their structure and content.
3. **Data Preparation**: Manually classify the images, labeling each bed according to the presence and posture of the cows.
4. **Modeling**: Build an image classification model using Convolutional Neural Networks (CNN) to detect whether a bed is occupied or empty and whether the cow is standing or lying down.
5. **Evaluation**: Validate the model's performance and fine-tune the parameters if needed.
6. **Deployment**: Generate reports and conclusions about the cows' bed usage patterns.

## Repository Structure

```bash
|-- dataset/
|   |-- classifier/
|   |   |-- original/            # Contains a sample of the aerial images (1920x1080 px) used for the classifier model.
|   |   |-- split/               # Contains the sample of the dataset split into train, test, and validation.
|   |   |-- transformed/         # Contains a sample of the cropped and classified images transformed for the classifier model.
|-- |-- bounding/
|   |   |-- annotations.csv      # Contains all coordenates of the bounding box data for every image in the dataset.
|   |   |-- original/            # Contains a sample of the aerial images (1920x1080 px) used for the bounding box model.
|   |   |-- split/               # Contains the sample of the dataset split into train, test, and validation.
|-- source/
|   |-- classifier.py            # Classification script for the original dataset.
|   |-- data_preparation.ipynb   # Code for data preparation.
|   |-- model_training.ipynb     # Notebook with the model training process.
|   |-- analysis.ipynb           # Analysis of results and findings.
|-- models/
|   |-- cnn_model.h5             # Trained classification model (CNN).
|-- results/
|   |-- reports/                 # Reports with analysis results.
|-- documentation/
|   |-- business/                # Documentation related to the business understanding phase.
|   |-- data/                    # Documentation related to data understanding and preparation phases.
|   |-- modeling/                # Documentation related to the modeling phase.
|   |-- evaluation/              # Documentation related to the evaluation phase.
|   |-- deployment/              # Documentation related to the deployment phase.
|-- README.md                    # Project description (this file).
