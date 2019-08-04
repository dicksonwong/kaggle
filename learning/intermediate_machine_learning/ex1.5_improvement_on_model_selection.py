# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")




import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
													  
													  



from sklearn.ensemble import RandomForestRegressor

n_estimators_list = [50, 75, 100, 125, 150, 175, 200]
max_depth_list = [-1, 5, 7, 10]

# Default value is 2
min_samples_split_list = [2, 15, 18, 20, 22, 25]

# Default is 'mse'
criterion_list = ['mse', 'mae']

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
# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

#models = [model_1, model_2, model_3, model_4, model_5]





from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

min_mae = score_model(models[0])

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    if mae < min_mae:
        min_mae = mae
        print(models_description[i])
		
		
		
Model 1 MAE: 24226
Model 2 MAE: 23854
Model 3 MAE: 24002
Model 4 MAE: 24300
Model 5 MAE: 23872
Model 6 MAE: 24076
Model 7 MAE: 24177
Model 8 MAE: 24448
Model 9 MAE: 24009
Model 10 MAE: 24133
Model 11 MAE: 24196
Model 12 MAE: 24369
Model 13 MAE: 25490
Model 14 MAE: 25005
Model 15 MAE: 25682
Model 16 MAE: 25297
Model 17 MAE: 25899
Model 18 MAE: 25174
Model 19 MAE: 26081
Model 20 MAE: 25235
Model 21 MAE: 25780
Model 22 MAE: 25691
Model 23 MAE: 26111
Model 24 MAE: 25611
Model 25 MAE: 23557
[50, 7, 2, 'mse']
Model 26 MAE: 23631
Model 27 MAE: 23509
[50, 7, 15, 'mse']
Model 28 MAE: 24088
Model 29 MAE: 23585
Model 30 MAE: 24175
Model 31 MAE: 23996
Model 32 MAE: 24077
Model 33 MAE: 23900
Model 34 MAE: 24419
Model 35 MAE: 24309
Model 36 MAE: 24729
Model 37 MAE: 24176
Model 38 MAE: 23768
Model 39 MAE: 23277
[50, 10, 15, 'mse']
Model 40 MAE: 23587
Model 41 MAE: 23740
Model 42 MAE: 24316
Model 43 MAE: 23902
Model 44 MAE: 24537
Model 45 MAE: 23979
Model 46 MAE: 24310
Model 47 MAE: 24501
Model 48 MAE: 24666
Model 49 MAE: 23492
Model 50 MAE: 23778
Model 51 MAE: 24253
Model 52 MAE: 24143
Model 53 MAE: 23866
Model 54 MAE: 24115
Model 55 MAE: 24134
Model 56 MAE: 24307
Model 57 MAE: 23983
Model 58 MAE: 24326
Model 59 MAE: 23847
Model 60 MAE: 24217
Model 61 MAE: 25510
Model 62 MAE: 24606
Model 63 MAE: 26438
Model 64 MAE: 25275
Model 65 MAE: 25955
Model 66 MAE: 25280
Model 67 MAE: 26076
Model 68 MAE: 25178
Model 69 MAE: 26171
Model 70 MAE: 25567
Model 71 MAE: 26302
Model 72 MAE: 25954
Model 73 MAE: 23580
Model 74 MAE: 23571
Model 75 MAE: 23817
Model 76 MAE: 24137
Model 77 MAE: 24027
Model 78 MAE: 24105
Model 79 MAE: 23791
Model 80 MAE: 24428
Model 81 MAE: 23846
Model 82 MAE: 24280
Model 83 MAE: 24369
Model 84 MAE: 24087
Model 85 MAE: 23773
Model 86 MAE: 23211
[75, 10, 2, 'mae']
Model 87 MAE: 23916
Model 88 MAE: 24292
Model 89 MAE: 23879
Model 90 MAE: 24356
Model 91 MAE: 23967
Model 92 MAE: 24187
Model 93 MAE: 24222
Model 94 MAE: 24000
Model 95 MAE: 24075
Model 96 MAE: 24601
Model 97 MAE: 23731
Model 98 MAE: 23469
Model 99 MAE: 24049
Model 100 MAE: 23906
Model 101 MAE: 24248
Model 102 MAE: 24179
Model 103 MAE: 24236
Model 104 MAE: 24190
Model 105 MAE: 24087
Model 106 MAE: 24135
Model 107 MAE: 24011
Model 108 MAE: 24122
Model 109 MAE: 25919
Model 110 MAE: 24545
Model 111 MAE: 25825
Model 112 MAE: 25202
Model 113 MAE: 26125
Model 114 MAE: 25126
Model 115 MAE: 25818
Model 116 MAE: 25382
Model 117 MAE: 26193
Model 118 MAE: 25248
Model 119 MAE: 26099
Model 120 MAE: 25614
Model 121 MAE: 23498
Model 122 MAE: 23711
Model 123 MAE: 23705
Model 124 MAE: 23872
Model 125 MAE: 23789
Model 126 MAE: 24264
Model 127 MAE: 23756
Model 128 MAE: 23976
Model 129 MAE: 23783
Model 130 MAE: 24374
Model 131 MAE: 24127
Model 132 MAE: 24412
Model 133 MAE: 23449
Model 134 MAE: 23252
Model 135 MAE: 23694
Model 136 MAE: 23553
Model 137 MAE: 23687
Model 138 MAE: 24170
Model 139 MAE: 23908
Model 140 MAE: 24083
Model 141 MAE: 23990
Model 142 MAE: 24141
Model 143 MAE: 23988
Model 144 MAE: 24523
Model 145 MAE: 23824
Model 146 MAE: 23441
Model 147 MAE: 23915
Model 148 MAE: 24170
Model 149 MAE: 23619
Model 150 MAE: 23869
Model 151 MAE: 24018
Model 152 MAE: 24042
Model 153 MAE: 24127
Model 154 MAE: 24357
Model 155 MAE: 24317
Model 156 MAE: 24309
Model 157 MAE: 25553
Model 158 MAE: 25004
Model 159 MAE: 25876
Model 160 MAE: 24982
Model 161 MAE: 25810
Model 162 MAE: 25454
Model 163 MAE: 25984
Model 164 MAE: 25263
Model 165 MAE: 25876
Model 166 MAE: 25522
Model 167 MAE: 26124
Model 168 MAE: 25589
Model 169 MAE: 23611
Model 170 MAE: 23525
Model 171 MAE: 23755
Model 172 MAE: 23753
Model 173 MAE: 23803
Model 174 MAE: 24043
Model 175 MAE: 23853
Model 176 MAE: 23994
Model 177 MAE: 24157
Model 178 MAE: 24209
Model 179 MAE: 24074
Model 180 MAE: 24291
Model 181 MAE: 23839
Model 182 MAE: 23240
Model 183 MAE: 23523
Model 184 MAE: 23719
Model 185 MAE: 23771
Model 186 MAE: 23957
Model 187 MAE: 23686
Model 188 MAE: 23900
Model 189 MAE: 23968
Model 190 MAE: 24364
Model 191 MAE: 24351
Model 192 MAE: 24431
Model 193 MAE: 23779
Model 194 MAE: 23485
Model 195 MAE: 23793
Model 196 MAE: 23871
Model 197 MAE: 23800
Model 198 MAE: 24073
Model 199 MAE: 23682
Model 200 MAE: 24151
Model 201 MAE: 24197
Model 202 MAE: 24211
Model 203 MAE: 24234
Model 204 MAE: 24629
Model 205 MAE: 25767
Model 206 MAE: 24751
Model 207 MAE: 25730
Model 208 MAE: 25204
Model 209 MAE: 26065
Model 210 MAE: 25172
Model 211 MAE: 25909
Model 212 MAE: 25147
Model 213 MAE: 26128
Model 214 MAE: 25428
Model 215 MAE: 26153
Model 216 MAE: 25490
Model 217 MAE: 23114
[150, 7, 2, 'mse']
Model 218 MAE: 23052
[150, 7, 2, 'mae']
Model 219 MAE: 23473
Model 220 MAE: 23850
Model 221 MAE: 23771
Model 222 MAE: 24176
Model 223 MAE: 23821
Model 224 MAE: 24221
Model 225 MAE: 23903
Model 226 MAE: 24167
Model 227 MAE: 24147
Model 228 MAE: 24344
Model 229 MAE: 23936
Model 230 MAE: 23432
Model 231 MAE: 23755
Model 232 MAE: 23702
Model 233 MAE: 23880
Model 234 MAE: 23910
Model 235 MAE: 23930
Model 236 MAE: 24278
Model 237 MAE: 24078
Model 238 MAE: 24175
Model 239 MAE: 24124
Model 240 MAE: 24334
Model 241 MAE: 23879
Model 242 MAE: 23422
Model 243 MAE: 23957
Model 244 MAE: 23657
Model 245 MAE: 23887
Model 246 MAE: 23929
Model 247 MAE: 24081
Model 248 MAE: 24013
Model 249 MAE: 24054
Model 250 MAE: 24259
Model 251 MAE: 24071
Model 252 MAE: 24409
Model 253 MAE: 25771
Model 254 MAE: 24837
Model 255 MAE: 25791
Model 256 MAE: 24967
Model 257 MAE: 25976
Model 258 MAE: 25119
Model 259 MAE: 26019
Model 260 MAE: 25268
Model 261 MAE: 26048
Model 262 MAE: 25637
Model 263 MAE: 26064
Model 264 MAE: 25575
Model 265 MAE: 23307
Model 266 MAE: 23167
Model 267 MAE: 23417
Model 268 MAE: 23859
Model 269 MAE: 23640
Model 270 MAE: 24095
Model 271 MAE: 23743
Model 272 MAE: 24194
Model 273 MAE: 24031
Model 274 MAE: 24307
Model 275 MAE: 24068
Model 276 MAE: 24412
Model 277 MAE: 23599
Model 278 MAE: 23161
Model 279 MAE: 23782
Model 280 MAE: 23655
Model 281 MAE: 23756
Model 282 MAE: 23780
Model 283 MAE: 23947
Model 284 MAE: 24111
Model 285 MAE: 23983
Model 286 MAE: 24307
Model 287 MAE: 24250
Model 288 MAE: 24547
Model 289 MAE: 23646
Model 290 MAE: 23507
Model 291 MAE: 23927
Model 292 MAE: 23706
Model 293 MAE: 23783
Model 294 MAE: 24146
Model 295 MAE: 23858
Model 296 MAE: 24220
Model 297 MAE: 23866
Model 298 MAE: 24342
Model 299 MAE: 24180
Model 300 MAE: 24431
Model 301 MAE: 25451
Model 302 MAE: 24798
Model 303 MAE: 25893
Model 304 MAE: 25317
Model 305 MAE: 25874
Model 306 MAE: 25327
Model 307 MAE: 25916
Model 308 MAE: 25417
Model 309 MAE: 26043
Model 310 MAE: 25199
Model 311 MAE: 26228
Model 312 MAE: 25469
Model 313 MAE: 23127
Model 314 MAE: 23512
Model 315 MAE: 23827
Model 316 MAE: 23858
Model 317 MAE: 23698
Model 318 MAE: 24152
Model 319 MAE: 23736
Model 320 MAE: 24152
Model 321 MAE: 23929
Model 322 MAE: 24115
Model 323 MAE: 24202
Model 324 MAE: 24693
Model 325 MAE: 23825
Model 326 MAE: 23120
Model 327 MAE: 23809
Model 328 MAE: 23782
Model 329 MAE: 23619
Model 330 MAE: 23962
Model 331 MAE: 23732
Model 332 MAE: 24231
Model 333 MAE: 24218
Model 334 MAE: 24260
Model 335 MAE: 24304
Model 336 MAE: 24417


# Define a model
my_model = RandomForestRegressor(n_estimators=150, max_depth=7, criterion='mae', random_state=0) # Your code here

# Check your answer
step_2.check()


# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)