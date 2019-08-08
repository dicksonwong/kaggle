# (Optional) Your code here
# Use this model to evaluate different methods:
model = RandomForestRegressor(n_estimators=150,min_samples_split=2,max_depth=11,criterion='mse' , random_state=0)

#model.fit(X_train, y_train)
#preds = model.predict(X_valid)
#return mean_absolute_error(y_valid, preds)

### Method 1: try dropping

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# Check your answers
model.fit(drop_X_train, y_train)
preds = model.predict(drop_X_valid)
print(mean_absolute_error(preds, y_valid))

### Method 2: Label Encoding

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

model.fit(label_X_train, y_train)
preds = model.predict(label_X_valid)
print(mean_absolute_error(preds, y_valid))

### Method 3: Try OH encoding on different number of cardinality-columns

# Columns that will be one-hot encoded
'''
print("method 3: OH encoding on different cardinalities")
for card in [1,2,3,4,5,6,7,9,10,20,30]:
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < card]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

    # Drop the columns in high_cardinality_cols
    OH_X_train = X_train.drop(high_cardinality_cols, axis=1)
    OH_X_valid = X_valid.drop(high_cardinality_cols, axis=1)

    # Apply one-hot encoder to each column with low cardinality
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(OH_X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(OH_X_valid[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = OH_X_train.index
    OH_cols_valid.index = OH_X_valid.index

    # Remove low_cardinality_cols columns (will replace with one-hot encoding)
    num_X_train = OH_X_train.drop(low_cardinality_cols, axis=1)
    num_X_valid = OH_X_valid.drop(low_cardinality_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    model.fit(OH_X_train, y_train)
    preds = model.predict(OH_X_valid)
    print("card ", card, mean_absolute_error(preds, y_valid))
'''
'''
17383.01919040809
17222.106241198708
17325.910582838133
17230.226132700776
17209.726133611883
17047.356268691445
17047.356268691445
17047.356268691445
17047.356268691445
17281.439898021385
17116.377429706215
17116.377429706215
'''

### Method 4: Try OH encoding on different number of cardinality-columns, but also label encode
###           the columns the remaining columns
from sklearn.impute import SimpleImputer

# Examine test data first for cols with missing values
train_columns_with_missing_vals = [column for column in X_train.columns if X_train[column].isnull().any()]
valid_columns_with_missing_vals = [column for column in X_valid.columns if X_valid[column].isnull().any()]
test_columns_with_missing_vals = [column for column in X_test.columns if X_test[column].isnull().any()]
print(len(train_columns_with_missing_vals), " ", len(valid_columns_with_missing_vals), " ", len(test_columns_with_missing_vals))

# We discover quickly that dropping the columns with missing values will not produce good results - in the 18000 range,
# rather than the 17000 - even on validation data.

# This is an issue that might be encountered frequently - if the test data contains columns that are missing values that
# is not addressed with a strategy, what do we do?  This is an issue because we may not encounter such a situation
# with the training data.

# In fact, there might be a even larger issue - we may have object columns with missing data in the test data.  That is,
# we need to encode the column but we are also missing data.  We cannot perform imputation on object columns, but
# we cannot encode it if we are missing values.  

# One strategy may be to replace missing values with "NaN" string and encode it from there, but it may not be a faithful
# representation.  On the other hand, for that particular test example, "NaN" may be a good representation if that
# particular entry does not exist (for example, GarageBlt shouldn't not contain an entry if there exists no garage).

# Hence, we may want to encode "NaN" as a value if we are to make the assumption that missing values == invalid entry
# for that row/column.  

# Furthermore, if the test/validation data does not share this problem of having missing values, then the particular
# model trained fit according to the values is not largely affected by the new encoded values created by encoding 
# "NaN".  (Unless we fit according to continuous features)

# Overall, without inspecting the test data, we could not assign a good strategy to each column.  (Of course, we
# could examine ALL the columns in the training data and assign a strategy IN CASE we ever encounter missing values)
# The strategy we will go with forward is:
# 1. replacing missing numerical variable values with 0
# 2. replacing missing categorial variables with "missing_value"
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


transformers = []
cols_with_missing_string_data = []
for column in cols_with_missing_vals:
    # Check X_train for type - has no missing data
    if not is_numeric_dtype(X_train[column]):
        imputer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing_value'))])
    else:
        imputer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))])
    transformers.append((column, imputer, [column]))
    
preprocessor = ColumnTransformer(transformers, remainder='passthrough')


'''
# We will add an empty row to training data, validation data so that we may encode the missing values
new_row = pd.Series(name='NameOfNewRow')

X_train.append(new_row)
X_valid.append(new_row)
X_test.append(new_row)
'''
preprocessor.fit(X_train)

# Simple imputation
imputed_X_train = pd.DataFrame(preprocessor.transform(X_train))
imputed_X_valid = pd.DataFrame(preprocessor.transform(X_valid))
imputed_X_test = pd.DataFrame(preprocessor.transform(X_test))
            
imputed_X_train.columns = X_train.columns                                               
imputed_X_valid.columns = X_valid.columns                                                
imputed_X_test.columns = X_test.columns

'''
# Drop all columns with missing values
dropped_X_train = X_train.drop(cols_with_missing_vals, axis=1)
dropped_X_valid = X_valid.drop(cols_with_missing_vals, axis=1)
dropped_X_test = X_test.drop(cols_with_missing_vals, axis=1)
'''

object_cols = [col for col in imputed_X_train.columns if imputed_X_train[col].dtype == "object"]
# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                    set(imputed_X_train[col]) == set(imputed_X_valid[col]) ]

# Notice that there are categorical values appearing in the test data that does not appear anywhere else in the training data
# For these, we will either 
# 1. drop the columns altogether
# or 2. perform label encoding on all the columns
imputed_X_all = pd.concat([imputed_X_train, imputed_X_valid, imputed_X_test])

'''
for col in imputed_X_all.columns:
    print(col, " ", imputed_X_all[col].dtype)
    print(imputed_X_all[col].unique())
'''

# Columns that will be one-hot encoded
print("method 3: OH encoding on different cardinalities + label encoding.  This is performed on just 'good' label columns")
for card in [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]:
    low_cardinality_cols = [col for col in object_cols if imputed_X_train[col].nunique() < card]
    label_encoded_cols = list(set(good_label_cols) - set(low_cardinality_cols))
    print("number of cols that can be label encoded: ", len(label_encoded_cols))

    # Columns that will be dropped from the dataset
    columns_to_be_dropped = list(set(object_cols) - (set(low_cardinality_cols+label_encoded_cols)))

    # Drop the columns in high_cardinality_cols
    encoded_X_train = imputed_X_train.drop(columns_to_be_dropped, axis=1)
    encoded_X_valid = imputed_X_valid.drop(columns_to_be_dropped, axis=1)
    encoded_X_test = imputed_X_test.drop(columns_to_be_dropped, axis=1)
    encoded_X_all = imputed_X_all.drop(columns_to_be_dropped, axis=1)

    # Apply one-hot encoder to each column with low cardinality
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Fit the encoder based on ALL values seen in all data
    OH_encoder.fit(encoded_X_all[low_cardinality_cols])
    
    OH_cols_train = pd.DataFrame(OH_encoder.transform(encoded_X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(encoded_X_valid[low_cardinality_cols]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(encoded_X_test[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = encoded_X_train.index
    OH_cols_valid.index = encoded_X_valid.index
    OH_cols_test.index = encoded_X_test.index

    # Remove low_cardinality_cols columns (will replace with one-hot encoding)
    num_X_train = encoded_X_train.drop(low_cardinality_cols, axis=1)
    num_X_valid = encoded_X_valid.drop(low_cardinality_cols, axis=1)
    num_X_test = encoded_X_test.drop(low_cardinality_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    encoded_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    encoded_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    encoded_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

    encoded_X_all = pd.concat([encoded_X_train, encoded_X_valid, encoded_X_test])

    # Label encode the columns that have yet to be encoded
    label_encoder = LabelEncoder()
    for col in label_encoded_cols:
        label_encoder.fit(encoded_X_all[col])
        encoded_X_train[col] = label_encoder.transform(encoded_X_train[col])
        encoded_X_valid[col] = label_encoder.transform(encoded_X_valid[col])
        encoded_X_test[col] = label_encoder.transform(encoded_X_test[col])
    model.fit(encoded_X_train, y_train)
    preds = model.predict(encoded_X_valid)
    print("card ", card, mean_absolute_error(preds, y_valid))
    
'''
17383.01919040809
17222.106241198708
number of cols that can be label encoded:  20
card  1 25016.84627243856
number of cols that can be label encoded:  20
card  2 25016.84627243856
number of cols that can be label encoded:  18
card  3 25103.143212304738
number of cols that can be label encoded:  15
card  4 24961.90688743544
number of cols that can be label encoded:  9
card  5 24585.48763752031
number of cols that can be label encoded:  4
card  6 24897.136013859817
number of cols that can be label encoded:  3
card  7 24852.089071684713
number of cols that can be label encoded:  2
card  9 23909.17448619052
number of cols that can be label encoded:  2
card  10 23606.476310140115
number of cols that can be label encoded:  2
card  11 21545.173421508123
number of cols that can be label encoded:  2
card  12 21545.173421508123
number of cols that can be label encoded:  1
card  13 21531.39766841269
number of cols that can be label encoded:  1
card  14 21531.39766841269
number of cols that can be label encoded:  1
card  15 21531.39766841269
number of cols that can be label encoded:  0
card  16 21484.283126446568
'''

'''
# Final output from the latest
model.fit(encoded_X_train,y_train)
preds_test = model.predict(encoded_X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
'''



### Method 4: Try OH encoding on different number of cardinality-columns, but also label encode
###           the columns the remaining columns



object_cols = [col for col in imputed_X_train.columns if imputed_X_train[col].dtype == "object"]


print("method 4: perform one hot encoding on low cardinality columns and perform label encoding otherwise.  We perform encoding on ALL object_cols")
for card in [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,20,25,30,36,40,46,50,51, 56, 58, 65, 66, 70, 76, 81, 99, 100]:
    low_cardinality_cols = [col for col in object_cols if imputed_X_train[col].nunique() < card]

	# This is the important part: previously, we only examined the good_label_cols as consideration
    # for label encoding, dropping all columns for which there are values in that column (for the validation or
    # test data) that do not appear in the training data
    label_encoded_cols = list(set(object_cols) - set(low_cardinality_cols))
    print("number of cols that can be label encoded: ", len(label_encoded_cols))

    # Columns that will be dropped from the dataset
    columns_to_be_dropped = list(set(object_cols) - (set(low_cardinality_cols+label_encoded_cols)))

    # Drop the columns in high_cardinality_cols
    encoded_X_train = imputed_X_train.drop(columns_to_be_dropped, axis=1)
    encoded_X_valid = imputed_X_valid.drop(columns_to_be_dropped, axis=1)
    encoded_X_test = imputed_X_test.drop(columns_to_be_dropped, axis=1)
    encoded_X_all = imputed_X_all.drop(columns_to_be_dropped, axis=1)

    # Apply one-hot encoder to each column with low cardinality
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # Fit the encoder based on ALL values seen in all data
    try:
        OH_encoder.fit(encoded_X_all[low_cardinality_cols])
    except:
        for col in low_cardinality_cols:
            print(encoded_X_train[col].unique())
            print(encoded_X_valid[col].unique())
            print(encoded_X_test[col].unique())
            print(col)
            print(encoded_X_all[col].unique())
            print(encoded_X_all[col].dtype)
        break
    OH_cols_train = pd.DataFrame(OH_encoder.transform(encoded_X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(encoded_X_valid[low_cardinality_cols]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(encoded_X_test[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = encoded_X_train.index
    OH_cols_valid.index = encoded_X_valid.index
    OH_cols_test.index = encoded_X_test.index

    # Remove low_cardinality_cols columns (will replace with one-hot encoding)
    num_X_train = encoded_X_train.drop(low_cardinality_cols, axis=1)
    num_X_valid = encoded_X_valid.drop(low_cardinality_cols, axis=1)
    num_X_test = encoded_X_test.drop(low_cardinality_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    encoded_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    encoded_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    encoded_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

    encoded_X_all = pd.concat([encoded_X_train, encoded_X_valid, encoded_X_test])

    # Label encode the columns that have yet to be encoded
    label_encoder = LabelEncoder()
    for col in label_encoded_cols:
        label_encoder.fit(encoded_X_all[col])
        encoded_X_train[col] = label_encoder.transform(encoded_X_train[col])
        encoded_X_valid[col] = label_encoder.transform(encoded_X_valid[col])
        encoded_X_test[col] = label_encoder.transform(encoded_X_test[col])
    model.fit(encoded_X_train, y_train)
    preds = model.predict(encoded_X_valid)
    print("card ", card, mean_absolute_error(preds, y_valid))


# Final output from the latest
model.fit(encoded_X_train,y_train)
preds_test = model.predict(encoded_X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

'''
method 3: OH encoding on different cardinalities + label encoding
number of cols that will be OH-encoded:  0
number of cols that can be label encoded:  60
card  1 16805.64340176728
number of cols that will be OH-encoded:  0
number of cols that can be label encoded:  60
card  2 16805.64340176728
number of cols that will be OH-encoded:  3
number of cols that can be label encoded:  57
card  3 16719.423081218567
number of cols that will be OH-encoded:  7
number of cols that can be label encoded:  53
card  4 16832.831467453034
number of cols that will be OH-encoded:  15
number of cols that can be label encoded:  45
card  10 16951.600390365795
number of cols that will be OH-encoded:  36
number of cols that can be label encoded:  24
'''
