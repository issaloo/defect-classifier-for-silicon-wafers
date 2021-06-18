# Detecting and Labeling Failure Areas on Silicon Wafers


* [Background](#background)
* [Project Overview](#project-overview)
* [Libraries & Dependencies](#libraries-&-dependencies)
* [Directory](#directory)
* [Project Workflow](#project-workflow)
    1. [Data Exploration](#data-exploration)
    2. [Data Preparation](#data-preparation)
    3. [Feature Engineering](#feature-engineering)
    4. [Modeling](#modeling)
    5. [Full Pipeline](#pipeline)

## Background
Semiconductor chips are essential to electronic devices such as cars, phones, and appliances. Chip fabrication begins with a silicon wafer and follows hundreds of processing steps. There are four main processing steps: deposition, etch (removal), patterning, and modification (of electrical properties). During processing, defects from multiple sources (e.g., tools, air) land on the wafer and impede electrical connections and performance of the chip. Wafer inspections are implemented between process steps to detect pattern defects, which drive root cause analyses and enable corrective action. When defect issues are fixed earlier (i.e., before more wafers are affected), yield increases and manufacturing cost per chip decreases. After many processing steps, the wafer is then cut into chips, tested, packaged, and sold.

## Project Overview
Pattern defect detection is time consuming and expensive, requiring trained engineers and technicians. An automated, accurate defect classifier can reduce manual misidentification and allow engineers and technicians to more quickly identify root cause. Two common types of pattern defects are streaks and focus spots (i.e., area with high density of defects). 

Defect classifiers were built through using two machine learning models (KNN, Random Forest) and are shown to have a moderate classification accuracy. The optimal defect classifier can be deployed, if desired.

## Libraries & Dependencies

Language
- Python 3.9.5

Environment
- Jupyter Lab/Notebook

Standard Modules
- glob
- joblib
- os
- pickle
- sys
- time

Third Party Modules
- colorama
- convex hull
- hdbscan
- imblearn
- IPython
- matplotlib
- numpy
- openpyxl
- pandas
- scipy
- seaborn
- sklearn

## Directory
There are three dataset folders that contain datasets from raw to model-ready and one folder that holds all output files. 

| Folder | Description |
| ----- | ----- |
| raw_dataset | Raw wafer scan data, identified by unique Wafer Scribe ID |
| labeled_dataset | Labeled wafer scan data, identified by unique Wafer Scribe ID |
| model_dataset | Cluster features (e.g., area, number of points) from wafer scan data, aggregated into a single file |
| outputs | Outputs from running code |

## Project Workflow
The workflow follows the programs listed below:

| Programs | Description |
| ----- | ----- |
| 1_Data_Exploration.ipynb | |
| 2_Data_Preparation.ipynb | |
| 3_1_Feature_Engineering.ipynb | |
| 4_1_Modeling_KNN.ipynb | |
| 4_2_Modeling_DecisionTrees(RF).ipynb | |
| 5_Model_Pipeline.ipynb | |


## Data Exploration

## Data Preparation

## Feature Engineering

## Modeling

### KNN

### Decision Trees (Random Forest)

## Pipeline