# Detecting and Labeling Failure Areas on Silicon Wafers


* [Background](#background)
* [Libraries & Dependencies](#libraries-and-dependencies)
* [Purpose](#purpose)


## Background
Semiconductor chips are essential to and used in electronic devices such as cars, phones, and appliances. Chip fabrication can be separated into four main processes: deposition, etch (removal), patterning, and modification (of electrical properties). Between each process, 

## Purpose


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


| Folder | Description |
| ----- | ----- |
| raw_dataset | Raw wafer scan data, identified by unique Wafer Scribe ID |
| labeled_dataset | Labeled wafer scan data, identified by unique Wafer Scribe ID |
| model_dataset | Cluster features (e.g., area, number of points) from wafer scan data, aggregated into a single file |
| outputs | Outputs from running code |


| Programs | Description |
| ----- | ----- |
| 1_Data_Exploration.ipynb | |
| 2_Data_Preparation.ipynb | |
| 3_1_Feature_Engineering.ipynb | |
| 4_1_Modeling_KNN.ipynb | |
| 4_2_Modeling_DecisionTrees(RF).ipynb | |
| 5_Model_Pipepline.ipynb | |



