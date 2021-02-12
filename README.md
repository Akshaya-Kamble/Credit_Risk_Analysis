# Credit Risk Analysis using supervised machine learning

### Overview of the Analysis 
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the data will be oversampled using the RandomOverSampler and SMOTE algorithms, and then undersample the data using the ClusterCentroids algorithm. Then,using a combinatorial approach of over- and undersampling using the SMOTEENN algorithm we can compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. After checking the models we can evaluate the performance of these models and recommend on whether they should be used to predict credit risk.

### Results
#### Deliverable 1 : Resampling Models to Predict Credit Risk
Using the knowledge of the imbalanced-learn and scikit-learn libraries,we will evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First,we will use the oversampling RandomOverSampler and SMOTE algorithms, and then we will use the undersampling ClusterCentroids algorithm. Using these algorithms, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.We will also use random state of 1 for each sampling algorithm to ensure consistency between tests.

#### A. Oversampling
We have compared two oversampling algorithms below to determine which algorithm results in the best performance.Both these algorithms have a similar balanced acuracy score.

#### 1. Naive Random Oversampling

The precision and recall are high for the low_risk group while precision and recall are low for the high_risk group

a. Balanced accuracy score :  65%
b. Precision score avg : 0.99
c. Recall score avg: 0.61 

![]()

#### 2. SMOTE Oversampling 
The precision,recall and F1 scores are similar to the Random Oversampling

a. Balanced accuracy score : 66%
b. Precision score avg : 0.99 
c. Recall score avg : 0.69

![]()

#### B. Undersampling
We have tested undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above.The balance accuracy score for the undersampling algorithm was less than the oversampling algorithms.

#### 1.Undersampling using ClusterCentroids resampler
The balanced accuracy score is lower than the oversampling algorithms.

a. Balanced accuracy score : 54%
b. Precision score avg : 0.99
c. Recall score avg : 0.42 

![]()

#### Deliverable 2 : Use the SMOTEENN algorithm to Predict Credit Risk
In the algorithm we will view the count of the target classes using Counter from the collections library,use the resampled data to train a logistic regression model,calculate the balanced accuracy score from sklearn.metrics,print the confusion matrix from sklearn.metrics and generate a classication report using the imbalanced_classification_report from imbalanced-learn.We will use a random state of 1 for the sampling algorithm to ensure consistency between tests

#### A.Combination (Over and Under) Sampling
We will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above.We will resample the data using the SMOTEENN algorithm.The balanced accuracy score was similar to the scores of the oversampling algorithms.

a. Balanced accuracy score : 66% 
b. Precision score avg : 0.99
c. Recall score avg : 0.69 

![]()

#### Deliverable 3 :Use Ensemble Classifiers to Predict Credit Risk
We will compare two ensemble algorithms to determine which algorithm results in the best performance.We will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier. Using the Algorithms we will train the model using the training data,calculate the balanced accuracy score from sklearn.metrics,print the confusion matrix from sklearn.metrics,generate a classication report using the imbalanced_classification_report from imbalanced-learn and for the Balanced Random Forest Classifier print the feature importance sorted in descending order along with the feature score.We will also use random state of 1 for each algorithm to ensure consistency between tests.

#### A.Balanced Random Forest Classifier
The balanced accuracy score is higher than the oversampling and undersampling models.

a. Balanced accuracy score : 78%
b. Precision score avg : 0.99
c. Recall score avg : 0.87

![]()

#### B.Easy Ensemble AdaBoost Classifier
The balanced accuracy score is highest in this algorithm and can be considered for predicting credit risk.

a. Balanced accuracy score : 93%
b. Precision score avg : 0.99
c. Recall score avg : 0.94

![]()

### Summary
The prefered model would be Easy Ensemble AdaBoost Classifier as it has the highest balance accuracy score of 93% amongst all the machine learning models. The average precision score for all the machine learning models is 0.99 and the recall score is highest for Easy Ensemble AdaBoost Classifier that is 0.94.