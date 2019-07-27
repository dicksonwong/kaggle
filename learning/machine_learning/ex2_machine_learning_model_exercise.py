# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")


# print the list of columns in the dataset to find the name of the prediction target
home_data.columns

# target variable
y = home_data.SalePrice

step_1.check()

# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# select data corresponding to features in feature_names
X = home_data[feature_names]

step_2.check()

# Review data
# print description or statistics from X
print(X)

# print the top few lines
print(X.head)

from sklearn.tree import DecisionTreeRegressor

#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X,y)

step_3.check()


# Use the model to predict selling price for training examples in X
predictions = iowa_model.predict(X)
print(predictions)
step_4.check()

''' Follow up:
If we compare the actual outputs compared to the predictions, these are the indices
for which the predictions were incorrect:
102 118911.0 118964
126 135875.0 128000
145 132500.0 130000
193 132500.0 130000
232 106250.0 94500
363 106250.0 118000
472 146500.0 148000
550 133750.0 140000
593 142000.0 140000
690 144433.33333333334 141000
721 144433.33333333334 143000
736 93200.0 93500
831 147576.0 151000
850 134000.0 131500
894 118911.0 118858
1088 132500.0 137500
1090 93200.0 92900
1364 147576.0 144152
1368 142000.0 144000
1421 133750.0 127500
1422 134000.0 136500
1431 135875.0 143750
1441 144433.33333333334 149300
1452 146500.0 145000

If we inspect this data, it is clear that for many training examples, the prediction
is exactly the same as the output.  

On the other hand, notice that a small subset of the predictions are off slightly, 
with a few outliers.  For example, example 736 and 1090 both have a prediction of
93200, with actual output 93500 and 92900.  This is explained by the fact that
when training a decision tree model, when there are multiple training examples
that correspond to different output values, the mean is taken as the prediction
value.

On the other hand, for a training example for which it has a unique set of values
for the set of features, then the prediction is exactly the output value for
this training example - this would explain that for many training examples, the
prediction is exactly equal to the output value for that training example.

In fact, if we print all duplicate rows (for features specified in X), then we get

  LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  \
102      7018       1979      1535         0         2             4   
126      4928       1976       958         0         2             2   
145      2522       2004       970       739         2             3   
193      2522       2004       970       739         2             3   
232      1680       1972       483       504         1             2   
363      1680       1972       483       504         1             2   
472      3675       2005      1072         0         1             2   
550      4043       1977      1069         0         2             2   
593      4435       2003       848         0         1             1   
690      4426       2004       848         0         1             1   
721      4426       2004       848         0         1             1   
736      8544       1950      1040         0         2             2   
831      3180       2005       520       600         2             2   
850      4435       2003       848         0         1             1   
894      7018       1979      1535         0         2             4   
1088     2522       2004       970       739         2             3   
1090     8544       1950      1040         0         2             2   
1364     3180       2005       520       600         2             2   
1368     4435       2003       848         0         1             1   
1421     4043       1977      1069         0         2             2   
1422     4435       2003       848         0         1             1   
1431     4928       1976       958         0         2             2   
1441     4426       2004       848         0         1             1   
1452     3675       2005      1072         0         1             2   

      TotRmsAbvGrd  
102              8  
126              5  
145              7  
193              7  
232              5  
363              5  
472              5  
550              4  
593              4  
690              3  
721              3  
736              6  
831              4  
850              3  
894              8  
1088             7  
1090             6  
1364             4  
1368             4  
1421             4  
1422             3  
1431             5  
1441             3  
1452             5  

Observe rows 736 and 1090 contain the same values, and we can confirm this to be true for the other examples.
'''

# Print rows for which the prediction is not equal to the real output
for i in range(0,len(y)):
    if y[i] != predictions[i]:
        print(i, y[i], predictions[i])
		
# Print all rows in X that has the exact same values for the features as another row
dupes = X[X.duplicated(keep=False)]
print(dupes)