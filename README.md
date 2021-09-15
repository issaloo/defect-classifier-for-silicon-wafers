# Defect Area Classifier for Silicon Wafers during Processing


* [Background](#background)
* [Project Overview](#project-overview)
* [Libraries and Dependencies](#libraries-and-dependencies)
* [Directories](#directories)
* [Techniques](#techniques)
* [Results](#results)

## Background
Semiconductor chips are essential to electronic devices such as cars, phones, and appliances. Chip fabrication begins with a silicon wafer and follows hundreds of processing steps to create intricate electronic circuits on the wafer. There are four main processing steps: deposition, etch (removal), patterning, and modification (of electrical properties). During processing, defects from multiple sources land on the wafer and eventually impede electrical connections and performance of the chip. Wafer inspections are implemented between process steps to detect these pattern defects, which drive root cause analyses and enable corrective action. When defect issues are fixed earlier (i.e., before more wafers are affected), yield increases and manufacturing cost per chip decreases. After finishing all processing steps, the wafer is then cut into chips, tested, packaged, and sold.

## Project Overview
Pattern defect detection is time consuming and expensive, requiring trained engineers and technicians. An automated, accurate defect classifier can reduce manual misidentification and allow engineers and technicians to more quickly identify root cause for the defect contributions. Two common types of pattern defects are streaks and focus spots (i.e., area with high density of defects).

Defect classifiers were built using two machine learning models (KNN, Random Forest), modeling techniques, and several data processing steps. The optimal defect classifier used the random forest model; however, further improvements to classification accuracy are needed before deployment.

## Libraries and Dependencies

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

## Directories
There are three dataset folders that contain datasets from raw to model-ready and one folder that holds all output files. 

| Folder | Description |
| ----- | ----- |
| raw_dataset | Raw wafer scan data, identified by unique Wafer Scribe ID |
| labeled_dataset | Labeled wafer scan data, identified by unique Wafer Scribe ID |
| model_dataset | Cluster features (e.g., area, number of points) from wafer scan data, aggregated into a single file |
| outputs | Outputs from running code |

The project workflow follows the programs listed below sequentially:
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

### Modeling
- Test Train Split
- K-Nearest Neighbors (KNN)
    - Normalization
        - Min Max Scaling
    - Hyperparameter Tuning
        - Elbow Method
- Random Forest
    - Hyperparameter Tuning
        - Class Weighting
        - Bootstrap Class Weighting
        - Randomized Grid Search
        - K-Fold Cross Validation
- Model Evaluation
    - Confusion Matrix
    - Classification Report
        - Precision
        - Recall
        - F1-Score

## Results
The entire dataset is highly imbalanced with classifications of 3366 nones, 960 focus spots, and 226 scratches. The dataset was split 70%-15%-15% for training, validation, and test sets, respectively. Using a tuned random forest model, the test data achieved these accuracy results:

|       | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| None | 0.95 | 0.95 | 0.95 | 505 |
| Focus Spot | 0.82 | 0.85 | 0.83 | 144 |
| Scratch | 0.57 | 0.47 | 0.52 | 34 |
| **Accuracy** | | | | |
| Macro Avg| 0.78 | 0.76 | 0.77 | 683 |

Because the dataset is imbalanced, increasing the accuracy metrics for scratches - the minority class - are prioritized. A precision of 57% and recall of 47% for scratch classification is moderate and requires higher accuracies for the classification model to be deployed. Further improvements can be made through training on other models (e.g., Linear Regression, SVM), utilizing different imbalanced dataset techniques, performing more hyperparameter tuning, and collecting more data.
