# Machine Learning Homework - Exoplanet Exploration
![exoplanets.jpg](Images/exoplanets.jpg)
#### Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

#### Use Jupyter Notebook, Pandas, Matplotlib, and Scikit-Learn to create machine learning models capable of classifying candidate exoplanets from the raw dataset.

###### Preprocess the Data
###### Preprocess the dataset prior to fitting the model.
###### Perform feature selection and remove unnecessary features.
###### Use MinMaxScaler to scale the numerical data.
###### Separate the data into training and testing data.
###### Tune Model Parameters
###### Use GridSearch to tune model parameters.
###### Tune and compare at least two different classifiers.



## Analysis
#### Deep Learning 1 Model
##### Downloaded Another data set cumulative.csv from kaggle as it has more elaborate columns.
##### The first step after reading the data to a dataframe is to decide which features to keep for the model.The aim wass to create machine learning models capable of classifying candidate exoplanets from the raw dataset.Soo we decided to ,load 21 columns from the data set for the Category CANDIDATE.
##### Then I classified the data based on the koi_plnt_tce_num.One hot encoding was not required in this case since the data was already classified based on the planet nums on filtering it by CANDIDATE.
##### I then split the data into 2 sets:Training & Test data sets.
##### Then i performed the Deep learning on the data set.
##### Got Loss: 0.0, Accuracy: 0.8386727571487427
##### Using GridSearchCV to tune the model's parameters and changing the grid parameters C and gamma and got {'C': 50, 'gamma': 0.0001}

#### Random Forest Model
##### The first step after reading the data to a dataframe is to decide which features to keep for the model. 
##### So i decided to perform a Random Forest Classifier search to do it.
##### I cleaned the data and divided it into 2 sets.
##### One with all the Numerical fields and the y values as the one with Koi_disposition.
##### I performed label encoding on the y field then.
##### This exoplanet data has koi_tce_plnt_num column that was not useful as a feature. It was just giving the planets numbers as grouping them.
##### I removed the features koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec the assumption was since their values are mostly 0's removing them would increase the accuracy of the model
##### After performing Random Classifier i got the sorted list of all the features by their importance and it turned out koi_score was the one with highest importance.
##### koi_score: A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition.
##### Then i decided to create clusters using make_blobs for the data and kmeans by using Koi-score as X-axis and Koi_disposition in y.


#### Deep learning Model
##### For this model, data cleaning and preprocessing steps were the same as Deep Learning-1 model except the fact that i decided to keep All the categories of Koi_disposition in this case and performing search on the raw data set.
##### After deciding which features to keep next step was assigning X and y values for the model to perform split data to get train and test data for the model.
##### Next step is to scale and normalize the data to create more accurate model that has less gap between data points so they all have acurate weights for the model. I used  MinMaxScaler to scale the data with deep learning model with a Loss: 0.258515864610672, Accuracy: 0.8998855948448181 after performing the deep learning.
##### Using GridSearchCV to tune the model's parameters, and changing C values, and increasing the number of iterations max_iter didn't improve scores sufficiently.
##### I got more accuracy in this Deep learning model as compared to the Deep Learning 1 Model and the RandomForest classifier is really helpful deciding the feature importances.