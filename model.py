import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV


file_path = './crop_yield_cleaned.csv'
yield_data_read = pd.read_csv(file_path)
yield_data_read.dropna(subset=['Yield_tons_per_hectare'], inplace=True)
yield_data = yield_data_read.sample(n=10000, random_state=1)

yield_data['Rainfall_Temperature_Ratio'] = yield_data['Rainfall_mm'] / yield_data['Temperature_Celsius']
yield_data['Soil_pH_Health'] = yield_data['Soil_pH'] * yield_data['Soil_Health']
yield_data['Rainfall_squared'] = yield_data['Rainfall_mm'] ** 2
yield_data['Temp_squared'] = yield_data['Temperature_Celsius'] ** 2

print(yield_data["Soil_pH"].describe())

# Update feature columns
feature_columns = ['Soil_pH', 'Soil_Health', 'Rainfall_mm', 'Temperature_Celsius', 
                  'Rainfall_Temperature_Ratio', 'Soil_pH_Health']
feature_columns = ['Crop', 'Soil_pH', 'Soil_Health', 'Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Irrigation_Used', 'Soil_Type'
]
X = yield_data[feature_columns]

X = pd.get_dummies(X, columns=['Crop', 'Soil_Type'], drop_first=True)
y = yield_data['Yield_tons_per_hectare']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# searching for params

param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), 
                          param_grid=param_grid, 
                          cv=5, 
                          n_jobs=-1, 
                        # verbose=2,
                          scoring='neg_mean_absolute_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

yield_model = grid_search.best_estimator_

# yield_model = RandomForestRegressor(n_estimators= 100, max_depth=10,  min_samples_leaf= 4, max_features=0.6, min_samples_split=2, n_jobs=-1)

yield_model.fit(X_train, y_train)
preds_val = yield_model.predict(X_test)

mae = mean_absolute_error(y_test, preds_val)
print(f"mae: {mae:.2f}")


mape = (abs(y_test - preds_val) / y_test).mean() * 100
print(f"MAPE: {mape:.2f}%")

final_model = RandomForestRegressor(
    **grid_search.best_params_, 
    random_state=0,
    n_jobs=-1
)
final_model.fit(X, y)


def estimate_crop_yield(
    crop, Soil_Type, Soil_pH=6.535328, Soil_Health=7.838000,
    Irrigation_Used=1, Fertilizer_Used=0, Rainfall_mm=551.866099, Temperature_Celsius=27.540676         #used mean values from the dataset
):
    input_dict = {
        'Soil_pH': [Soil_pH],
        'Soil_Health': [Soil_Health],
        'Rainfall_mm': [Rainfall_mm],
        'Temperature_Celsius': [Temperature_Celsius],
        'Fertilizer_Used': [Fertilizer_Used],
        'Irrigation_Used': [Irrigation_Used],
        'Crop': [crop],
        'Soil_Type': [Soil_Type]
    }

    input_df = pd.DataFrame(input_dict)
    input_df = pd.get_dummies(input_df, columns=['Crop', 'Soil_Type'], drop_first=True)

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[X.columns]

    prediction = final_model.predict(input_df)[0]
    return prediction



def estimate_best_crop():
    feature_means = X.mean()
    crops = yield_data['Crop'].unique()
    crops_estimations = {}
    
    for crop in crops:
        estimation = estimate_crop_yield(Soil_Type,crop ,Soil_pH, Soil_Health, Irrigation_Used ,Fertilizer_Used,Rainfall_mm, Temperature_Celsius) 
        crops_estimations[crop] = estimation
    return crops_estimations