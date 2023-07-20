# Human_Activity_Recognition
In this project, you'll learn different machine-learning algorithms for recognizing/predicting human activity.

## Workflow
### Step 1: Dataset
Here you can access the data using this link https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset
The data is prepared with the help of 30 people using smartphones.
Features in data are collected from the accelerometer(linear) and gyro meter(angular).
file with the name Inertial signals is the raw file, which we can directly use for the model in the neural network. 

#### 1.1 Data Cleaning
Checking for duplicate/ repetitive features.
Checking for NA/NaN value.

#### 1.2 Checking for data imbalance
Plotting the bar graphs for each subject and their respective collected data.

#### 1.3 Removing symbols
The symbol(-,(),_) in the names of features is get removed.

### Step 2: Exploratory Data Analysis
Drawing some graphs(line and Boxplot) using a single feature as a variable (univariate analysis). The graphs show the noticeable result in static and dynamic activities. Some features are suitable for only one activity.
Also, Here we use the t-SNE, which gives better visualization of the Activities. But still, sitting and standing activity shows one cluster or non-separable region.

### Step 3: Classical Machine Learning Model
Before going into modeling, we defined some functions to make evaluation easy (e.g., plot_confusion_matrix(), print_grid_search_attributes(), etc.)
Here we perform the following models
#### 3.1 Logistic regression 
#### 3.2 Linear SVC (Support Vector Classifier)
#### 3.3 Kernel SVM
#### 3.4 Decision Trees
#### 3.5 Random Forest
#### 3.6 Gradient Boosted Decision Trees 

### Step 4: Comparing the models
Comparing all model performances.

### Step 5: LSTM: Large Short-Term Memory (Sequential deep learning model)
Here we apply a simple LSTM model on raw data(Inertial Signals)


##### Library: python, pandas, numpy, tensorflow, keras, sci-kit learn
