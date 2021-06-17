# Detecting and Labeling Failure Areas on Silicon Wafers


* [Background](#background)
* [Purpose](#purpose)
* [Libraries & Dependencies](#libraries-&-dependencies)
* [Folders](#folders)
* [Project Workflow](#project-workflow)
    1. [Data Exploration](#data-exploration)
    2. [Data Preparation](#data-preparation)
    3. [Feature Engineering](#feature-engineering)
    4. [Modeling](#modeling)
    5. [Pipeline](#pipeline)

## Background
Semiconductor chips are essential to electronic devices such as cars, phones, and appliances. Chip fabrication begins with a silicon wafer and requires four main processes: deposition, etch (removal), patterning, and modification (of electrical properties). Wafer inspections are implemented between process steps to detect physical and pattern defects, which drive root cause analyses for quality issues and manufacturing corrections. The resulting actions from the information gained increase yield and decrease manufacturing cost per chip. After iterations of varied processing steps, the wafer is then cut into chips, tested, packaged, and sold.

## Purpose
....

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

## Folders
| Folder | Description |
| ----- | ----- |
| raw_dataset | Raw wafer scan data, identified by unique Wafer Scribe ID |
| labeled_dataset | Labeled wafer scan data, identified by unique Wafer Scribe ID |
| model_dataset | Cluster features (e.g., area, number of points) from wafer scan data, aggregated into a single file |
| outputs | Outputs from running code |

## Project Workflow
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