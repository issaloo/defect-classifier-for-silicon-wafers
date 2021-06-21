# Detecting and Labeling Failure Areas on Silicon Wafers


* [Background](#background)
* [Project Overview](#project-overview)
* [Libraries & Dependencies](#libraries-&-dependencies)
* [Directory](#directory)
* [Project Workflow](#project-workflow)

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

The project workflow follows the programs listed below:
| Programs | Description |
| ----- | ----- |
| 1_Data_Exploration.ipynb | Understand data structure, analyze data relationships, and visualize data with graphs|
| 2_Data_Preparation.ipynb | Evaluate clustering algorithms and implement semi-automated labeling program; Output: labeled_dataset |
| 3_1_Feature_Engineering.ipynb | Create features at the wafer (e.g., number of defects on wafer) and the cluster level (e.g., number of defects of cluster), aggregate processed data for modeling; Output: model_dataset |
| 3_2_Feature_Exploration.ipynb | Analyze feature relationships, visualize data with graphs, evaluate correlation between features |
| 4_1_Modeling_KNN.ipynb | Fit and evaluate processed data with K-Nearest Neighbors model |
| 4_2_Modeling_RandomForest.ipynb | Fit and evaluate processed data with Random Forest model |
| 5_Model_Pipeline.ipynb | Best model re-trained and end-to-end prediction pipeline from data input to processing to prediction; Output: gross_failure_classifier.sav |


## Techniques
### Data Processing
- Convex Hull Algorithm
- Pivot Tables

### Data Modeling
- K-Nearest Neighbors (KNN)
    - Normalization
        - MinMaxScaler
    - Hyperparameter Tuning
        - Elbow Method
- Random Forest
    - gs
- Model Evaluation


## Results