# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex2 import *
print("Setup Complete")


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# dytypes:
# selects columns of specified type (exclude 'object' means we only take columns with numerical values)





# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


'''
(1168, 36)
LotFrontage    212
MasVnrArea       6
GarageYrBlt     58
dtype: int64
'''




# Fill in the line below: How many rows are in the training data?
num_rows = X_train.shape[0]
print(X_train.shape)
print(X_train.columns)

columns_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]

# Fill in the line below: How many columns in the training data
# have missing values?
num_cols_with_missing = len(columns_with_missing_vals)

# Fill in the line below: How many missing entries are contained in 
# all of the training data?
tot_missing = X_train.isnull().sum().sum()

# Check your answers
step_1.a.check()


'''
Considering your answers above, what do you think is likely the best approach to dealing with the missing values?

Only three columns have missing values; of those three columns, there are a total of 212, 6, and 58 entries
missing.  Dropping the entire column because of a missing value would lose us a lot of information (in fact,
there are 1168 rows in total); hence, the best approach in this case would be to imputate values (whether
by method 2 or method 3).  Because there aren't too many values missing, we would likely not gain too much
information by adding column to indicate which rows were missing values (method 3).  More importantly,
the columns with missing information are:

['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

It seems likely that these columns with missing information are just missing values because it was not
provided; hence, knowing that these were missing information does not (at first glance, without any
analysis into the other values in the rows with missing entries) add any more information to the model
that would likely help build a better model.

Realistically, we would just build both models and compare mae.
'''



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Step 2: Drop columns with missing values

# Fill in the line below: get names of columns with missing values
cols_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(columns=cols_with_missing_vals, axis=1)
reduced_X_valid = X_valid.drop(columns=cols_with_missing_vals, axis=1)

# Check your answers
step_2.check()


'''
MAE (Drop columns with missing values):
17837.82570776256
'''



# Step 3: Imputer
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer() # Your code here

# Fit the imputer with the training data; this way, we will fill in missing values in
# both the training data and validation data with the means fitted for the training
# data; the validation data assumes imputed values corresponding to means calculated
# via the training data
my_imputer.fit(X_train)
imputed_X_train = pd.DataFrame(my_imputer.transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Check your answers
step_3.a.check()


print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

'''
MAE (Imputation):
18062.894611872147
'''

'''
Compare the MAE from each approach.  Does anything surprise you about the results?  Why do you think one approach performed better than the other?

It is a bit surprising to see that the model with imputed values performed slightly lower than the one simply dropping the columns.  This may
perhaps indicate that substituting the mean values for missing values does not perform well.  In particular, we replaced the values for these
columns:

['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

The most questionable choice here is GarageYrBlt - if we simply replaced missing values with the mean value, we would perhaps get odd combinations
of HouseYrBlt with GarageYrBlt;hence, there may be better choices than the mean for these columns.

On the other hand, in the previous example, the columns with missing information are:

(10864, 12)
Car               49
BuildingArea    5156
YearBuilt       4307
dtype: int64

In the example, the model with imputed values performed better, suggesting that mean values are reasonable replacements for the missing values.
'''





# Method 3: Try extended imputation - add a column indicating which vals missing

# New DataFrames with added columns
X_train_with_new_cols = X_train.copy()
X_valid_with_new_cols = X_valid.copy()

# Add columns to new dataframes that indicate whether specific column is missing
cols_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]
for column in cols_with_missing_vals:
    X_train_with_new_cols[column + "_is_missing_val"] = X_train[column].isnull()
    X_valid_with_new_cols[column + "_is_missing_val"] = X_valid[column].isnull()

# Imputation for missing values
my_imputer = SimpleImputer()
my_imputer.fit(X_train_with_new_cols)

# Preprocessed training and validation features
final_X_train = pd.DataFrame(my_imputer.transform(X_train_with_new_cols))
final_X_valid = pd.DataFrame(my_imputer.transform(X_valid_with_new_cols))

print("MAE (Extended Imputation):")
print(score_dataset(final_X_train, final_X_valid, y_train, y_valid))

# Check your answers
step_4.a.check()

'''
MAE (Extended Imputation):
18148.417180365297
'''





# Method 4: Try different imputation methods

# Note that constant will use fill_value, which is 0 by default
strategies = ["mean", "median", "most_frequent", "constant"]
imputation_methods = ["simple", "extended"]

'''
Notice that in *almost* all cases, the simple method performs better than the extended method;
the only exception here is the "constant" method, which uses a fill_value=0 approach, which is
clearly not optimal.

On the other hand, median + simple seems to perform the best (even better than removing columns
with missing values).  We will use this approach and test with different forests.
'''




# Interestingly, the best performing models are:
# MAE   ['simple', 'mean']       [100, 9, 3, 'mse']
# 17668.079349940057
# MAE   ['simple', 'constant']       [100, 9, 3, 'mse']
# 17644.745974071906

'''
In this case, even a simple imputation of constant value fill_value=0 works quite well.

Perhaps with smaller values of n_estimators, simpler methods such as simple+mean and simple+constant
work better; on the other hand, with larger values of n_estimators, more complicated methods such as
constant+extended work slightly better.

In fact, even for a larger value of n_estimators=160, simple+constant performed quite well:

MAE Strategy= constant Method= simple Model: [160, 8, 2, 'mse']
17777.157207489803

Let us inspect this a bit further.
If we print all the rows where GarageYrBlt is undefined AND GarageArea != 0, then notice there are
no such rows.

That is to say, the reason why the GarageYrBlt is missing is because there is no garage.  In this case,
it may actually be reasonable to give the GarageYrBlt an invalid value = 0.

On the other hand, it is also intuitive to use imputation fill_value of YearBuilt, as a missing
Garage could be thought vacuously built the same year as the house.

Now, if we analyze the rows, we find the following:
1. There are garages (with non-NaN values for GarageYrBlt) that has different year than YearBuilt
2. There are NO rows with "LotArea" == 0.
3. There do exists rows with "MasVnrArea" == 0.


Intuitively, these are the steps that may work well:
1.  Replacing missing values with GarageYrBlt with the same value in the row for YearBlt.
2.  Replacing missing values for LotArea with 0.
3.  Replacing MasVnrArea with 0 or the mean.

We can try a combination of these on the following models that seemed to work quite well.

Some of the best models are:
MAE   ['mean', 'mean', 0, 'extended']       [150, 9, 3, 'mse']
17384.477660980265
MAE   ['mean', 0, 'mean', 'simple']       [100, 9, 2, 'mse']
17535.499142318607

MAE   ['mean', 'mean', 'mean', 'extended']       [150, 9, 3, 'mse']
17687.728729952654
MAE   ['mean', 0, 0, 'simple']       [150, 9, 3, 'mse']
17696.25467731903

It is worth noting at this point that models with criterion='mae' don't seem to perform that well
compared to models with cerion='mse'.  It may be worth it to simply skip these models.
'''

# Notice that we are getting different maes for different models+training_sets.
# There is some randomness in training random forests.  The following code will fit model at least 10 times before calculating avg seen

'''
Some general observations:
1. mean+mean+mean imputation almost always performs worse
2. sweet spot for depth seems to be around 10-11, which seems to be one of the largest factors in model accuracy
   This observation holds for n_estimators between 100 to 175
3. After running many tests with different model+imputation method, it seems pretty clear that the imputation method
   does not make too large of a difference, although it seems like some of the most consistent combinations that tend
   to outperform other ones are 'mean' + 0 + 'mean' (for the three variables) - at the 17.7k range with a good model
   (best at n_estimators = 175, depth = 11, min_samples = 2, 'mse')
  
   Some other combinations also perform similarly ('most_frequent' or 'median') so similarly well, but it appears 
   that choosing particular combinations of imputations will perform better (e.g. 'mean' + 0 'mean' work better
   than choosing a single strategy across all columns)

   It may be worth trying different combinations in the future of 'most_frequent', 'mean', 'median', 0, etc.
   across different columns.
'''

########################### Final version
###########################

# Method 5: Try combinations of different random forest models + different imputation methods
# Note that in the previous exercise, we discovered that for this particular set of data, 
# the combination 150, 7, 2, "mae" worked quite well for a data set that removed columns
# with missing data.

# Extended Imputation Method:
# New DataFrames with added columns
X_train_with_new_cols = X_train.copy()
X_valid_with_new_cols = X_valid.copy()

# Add columns to new dataframes that indicate whether specific column is missing
cols_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]
for column in cols_with_missing_vals:
    X_train_with_new_cols[column + "_is_missing_val"] = X_train[column].isnull()
    X_valid_with_new_cols[column + "_is_missing_val"] = X_valid[column].isnull()
    
# Imputation for missing values
my_imputer = SimpleImputer()
my_imputer.fit(X_train_with_new_cols)

# Preprocessed training and validation features
final_X_train = pd.DataFrame(my_imputer.transform(X_train_with_new_cols))
final_X_valid = pd.DataFrame(my_imputer.transform(X_valid_with_new_cols))

print("MAE (Extended Imputation):")
print(score_dataset(final_X_train, final_X_valid, y_train, y_valid))



# Build different forest models
# Random forest models
n_estimators_list = [175,150,125,100]
max_depth_list = [11,10,9,8,7]

# Default value is 2
min_samples_split_list = [2, 3]

# Default is 'mse'
criterion_list = ['mse','mae']

models = []
models_description = []

# Create a model taking combinations of parameters from the above lists
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for min_samples in min_samples_split_list:
            for criterion in criterion_list:
                models_description.append([n_estimators, max_depth, min_samples, criterion])
                if max_depth == -1:
                    models.append(RandomForestRegressor(n_estimators=n_estimators,
                                                        min_samples_split=min_samples,
                                                        criterion=criterion))
                else:
                    models.append(RandomForestRegressor(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        min_samples_split=min_samples,
                                                        criterion=criterion))

# Note that constant will use fill_value, which is 0 by default
#strategies = ["mean", "most_frequent", "constant", "median"]
strategies = ["most_frequent", "median"]
imputation_methods = ["extended", "simple"]

final_train_data = []
final_valid_data = []
final_test_data = []
imputation_strategies = []

for strategy in strategies:
    for imputation_method in imputation_methods:
        # New DataFrames with possibly added columns
        X_train_with_new_cols = X_train.copy()
        X_valid_with_new_cols = X_valid.copy()
        X_test_with_new_cols = X_test.copy()
        
        # Add new columns in method is "extended"
        if imputation_method == "extended":
            # Add columns to new dataframes that indicate whether specific column is missing
            cols_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]
            for column in cols_with_missing_vals:
                X_train_with_new_cols[column + "_is_missing_val"] = X_train[column].isnull()
                X_valid_with_new_cols[column + "_is_missing_val"] = X_valid[column].isnull()
                X_test_with_new_cols[column + "_is_missing_val"] = X_test[column].isnull()
        # Imputation for missing values
        my_imputer = SimpleImputer(strategy=strategy, fill_value=0)
        my_imputer.fit(X_train_with_new_cols)
        
        # Preprocessed training and validation features
        final_train_data.append(pd.DataFrame(my_imputer.transform(X_train_with_new_cols)))
        final_valid_data.append(pd.DataFrame(my_imputer.transform(X_valid_with_new_cols)))
        final_test_data.append(pd.DataFrame(my_imputer.transform(X_test_with_new_cols)))
        imputation_strategies.append([imputation_method, strategy])

        # Fix column names
        final_train_data[-1].columns = X_train_with_new_cols.columns
        final_valid_data[-1].columns = X_valid_with_new_cols.columns
        final_test_data[-1].columns = X_test_with_new_cols.columns
        #print("MAE (Imputation); Strategy= " + strategy + " Method= " + imputation_method)
        #print(score_dataset(final_X_train, final_X_valid, y_train, y_valid))

        # Test different random forests models
        
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

impute_garage_yr_blt_strategy = ["mean", 0]
impute_lot_area_strategy = ["mean", 0]
impute_mas_vnr_area = ["mean", 0]

preprocessors = []
imputation_combinations = []
for garage_strategy in impute_garage_yr_blt_strategy:
    if garage_strategy == "mean":
        imputer_garage = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
    elif garage_strategy == "0":
        imputer_garage = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant", fill_value=0))])
                                          
    for lot_area_strategy in impute_lot_area_strategy:
        if lot_area_strategy == "mean":
            imputer_lotarea = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
        elif lot_area_strategy == "0":
            imputer_lotarea = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant", fill_value=0))])
                                               
        for mas_vnr_area_strategy in impute_mas_vnr_area:
            if mas_vnr_area_strategy == "mean":
                imputer_vnr = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
            elif mas_vnr_area_strategy == "0":
                imputer_vnr = Pipeline(steps=[('imputer', SimpleImputer(strategy="constant", fill_value=0))])

            imputation_combinations.append([garage_strategy, lot_area_strategy, mas_vnr_area_strategy])
            preprocessors.append(ColumnTransformer(transformers=[('imputer_garage',
                                                                  imputer_garage,
                                                                  ["GarageYrBlt"]),
                                                                 ('imputer_lotarea',
                                                                  imputer_lotarea,
                                                                  ["LotFrontage"]),
                                                                 ('imputer_vnr',
                                                                  imputer_vnr,
                                                                  ["MasVnrArea"])],
                                                   remainder='passthrough'))

                                
#final_train_data = []
#final_valid_data = []
#imputation_strategies = []
for i in range(0, len(preprocessors)):
    preprocessor = preprocessors[i]
    for imputation_method in imputation_methods:
        # New DataFrames with possibly added columns
        X_train_with_new_cols = X_train.copy()
        X_valid_with_new_cols = X_valid.copy()
        X_test_with_new_cols = X_test.copy()
        
        # Add new columns in method is "extended"
        if imputation_method == "extended":
            # Add columns to new dataframes that indicate whether specific column is missing
            cols_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]
            for column in cols_with_missing_vals:
                X_train_with_new_cols[column + "_is_missing_val"] = X_train[column].isnull()
                X_valid_with_new_cols[column + "_is_missing_val"] = X_valid[column].isnull()
                X_test_with_new_cols[column + "_is_missing_val"] = X_test[column].isnull()

        # Fit Preprocessor:
        preprocessor.fit(X_train_with_new_cols)
        # Impute data - returns numpy array
        # Preprocessed training and validation features
        final_train_data.append(pd.DataFrame(preprocessor.transform(X_train_with_new_cols)))
        final_valid_data.append(pd.DataFrame(preprocessor.transform(X_valid_with_new_cols)))
        final_test_data.append(pd.DataFrame(preprocessor.transform(X_test_with_new_cols)))

        # Fix column names
        final_train_data[-1].columns = X_train_with_new_cols.columns
        final_valid_data[-1].columns = X_valid_with_new_cols.columns
        final_test_data[-1].columns = X_test_with_new_cols.columns
        
        impute_strat = imputation_combinations[i].copy()
        impute_strat.append(imputation_method)
        imputation_strategies.append(impute_strat)
number_iter = 10
best_mae = 999999

'''
for i in range(0,len(models)):
    for j in range(0, len(final_train_data)):
        # skip 100 and 125 and 150 since we have an idea of those already
        #if models_description[i][0] < 175:
            #continue
        # also skip depth = 7
        # skip cases where data imputation for all three columns uses mean
        #if imputation_strategies[j][0] == "mean" and imputation_strategies[j][1] == "mean" and imputation_strategies[j][2] == "mean":
            #continue
        if models_description[i][1] < 10:
            continue
        model = models[i]
        final_X_train = final_train_data[j]
        final_X_valid = final_valid_data[j]


        total_mae = 0
        for k in range(0, number_iter):
            model.fit(final_X_train, y_train)
            
            # Get validation predictions and MAE
            preds_valid = model.predict(final_X_valid)
            total_mae += mean_absolute_error(y_valid, preds_valid)
        
        avg_mae = total_mae / number_iter
        if avg_mae < best_mae:
            print("i: " + str(i) + " j: " + str(j))
            print("MAE   " + str(imputation_strategies[j]) + "       " + str(models_description[i]))
            print(avg_mae)
            best_mae = avg_mae
'''
# Check your answers
step_4.a.check()



best_model_index = 0
best_training_data_index = 9

#Define and fit model
model = models[best_model_index]
final_X_train = final_train_data[best_training_data_index]
final_X_valid = final_valid_data[best_training_data_index]
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your appraoch):")
print(mean_absolute_error(y_valid, preds_valid))





# Fill in the line below: preprocess test data
final_X_test = final_test_data[best_training_data_index]

# Some of the columns in the test data may still have missing data
# Apply a 'mean' method to the rest of these
my_imputer = SimpleImputer(strategy='mean')
my_imputer.fit(final_X_test)
final_X_test = pd.DataFrame(my_imputer.transform(final_X_test))

# Fill in the lines below: imputation removed column names; put them back
final_X_test.columns = X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

step_4.b.check()



# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

