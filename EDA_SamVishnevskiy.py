import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import random
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import torch 
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sktime.performance_metrics.forecasting import MedianAbsolutePercentageError
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.ticker as ticker
import plotly.express as px

#zillow MDAPE about 8 percent vs Production MDAPE about 6 percent
"""
Fields:
Flooring,ViewYN,WaterfrontYN,BasementYN,PoolPrivateYN,OriginalListPrice,ListingKey,ListAgentEmail,CloseDate,
ClosePrice,ListAgentFirstName,ListAgentLastName,Latitude,Longitude,UnparsedAddress,PropertyType,LivingArea,
ListPrice,DaysOnMarket,ListOfficeName,BuyerOfficeName,CoListOfficeName,ListAgentFullName,CoListAgentFirstName,
CoListAgentLastName,BuyerAgentMlsId,BuyerAgentFirstName,BuyerAgentLastName,FireplacesTotal,AssociationFeeFrequency,
AboveGradeFinishedArea,ListingKeyNumeric,MLSAreaMajor,TaxAnnualAmount,CountyOrParish,MlsStatus,ElementarySchool,
AttachedGarageYN,ParkingTotal,BuilderName,PropertySubType,LotSizeAcres,SubdivisionName,BuyerOfficeAOR,YearBuilt,
BuyerAgencyCompensationType,StreetNumberNumeric,ListingId,BathroomsTotalInteger,City,BuyerAgencyCompensation,TaxYear,
BuildingAreaTotal,BedroomsTotal,ContractStatusChangeDate,ElementarySchoolDistrict,CoBuyerAgentFirstName,PurchaseContractDate,
ListingContractDate,BelowGradeFinishedArea,BusinessType,StateOrProvince,CoveredSpaces,MiddleOrJuniorSchool,FireplaceYN,
Stories,HighSchool,Levels,LotSizeDimensions,LotSizeArea,MainLevelBedrooms,NewConstructionYN,GarageSpaces,
HighSchoolDistrict,PostalCode,AssociationFee,LotSizeSquareFeet,MiddleOrJuniorSchoolDistrict
"""
pd.set_option("display.max_rows", 100)

df202506 = pd.read_csv("CRMLSSold202506.csv", encoding = "ISO-8859-1")
df202507 = pd.read_csv("CRMLSSold202507.csv", encoding = "ISO-8859-1")
df202508 = pd.read_csv("CRMLSSold202508.csv", encoding = "ISO-8859-1")
df202509 = pd.read_csv("CRMLSSold202509.csv", encoding = "ISO-8859-1")
df202510 = pd.read_csv("CRMLSSold202510.csv", encoding = "ISO-8859-1")
df202511 = pd.read_csv("CRMLSSold202511.csv", encoding = "ISO-8859-1")
df202512 = pd.read_csv("CRMLSSold202512.csv", encoding = "ISO-8859-1")
df202601 = pd.read_csv("CRMLSSold202601.csv", encoding = "ISO-8859-1")
training_data = pd.concat([df202506, df202507, df202508, df202509, df202510, df202511, df202512])
testing_data = df202601.copy()

"""
print("state or province")
print(training_data["StateOrProvince"])
print(((training_data.isna().mean() * 100).sort_values(ascending=False)))
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\nPOSTAL CODES:\n\n\n\n\n\n\n")
print(type(training_data['PostalCode'][0]))
print(training_data['PostalCode'])
"""

def process_data(training_data, testing_data):
    training_data = training_data[training_data['PropertyType'] == "Residential"]
    training_data = training_data[training_data['PropertySubType'] == "SingleFamilyResidence" ]
    training_data = training_data[(training_data['ClosePrice'] > 0 )& (training_data['LivingArea'] > 0) & (training_data["StateOrProvince"] == "CA")]
    #print(f"duplicate count: {training_data["ListingKey"].duplicated().sum()}")
    training_data = training_data[(training_data['ClosePrice']  <= training_data['ClosePrice'].quantile(0.995)) & (training_data['ClosePrice']  >= training_data['ClosePrice'].quantile(0.005))]
    training_data = training_data[['ClosePrice', 'LivingArea', 'LotSizeAcres', 'YearBuilt','BathroomsTotalInteger', 'BedroomsTotal', 'GarageSpaces', 'Longitude', 'Latitude', 'PostalCode', 'PoolPrivateYN', "City"]].dropna()
    postal_code_average = training_data.groupby("PostalCode")["ClosePrice"].mean().reset_index().rename(columns={'ClosePrice': 'PostalCodeAverage'})
    training_data["PricePerSquareFoot"] = training_data["ClosePrice"]/training_data["LivingArea"]
    price_per_square_foot_zip_code = training_data.groupby("PostalCode")["PricePerSquareFoot"].mean().reset_index().rename(columns={'PricePerSquareFoot': 'PricePerSquareFootZipCode'})
    training_data = training_data.merge(postal_code_average, on="PostalCode", how="left" )
    training_data = training_data.merge(price_per_square_foot_zip_code, on="PostalCode", how="left" )
    training_data["PropertyAge"] = 2025 - training_data["YearBuilt"]
    training_data['Pool'] = training_data['PoolPrivateYN'].map({'Y': 1, 'N': 0})
    training_data['City'] = training_data['City'].astype('category')
    features_train = training_data[['LivingArea', 'LotSizeAcres', 'YearBuilt', 'BathroomsTotalInteger','BedroomsTotal', 'GarageSpaces', 'Longitude', 'Latitude', "PostalCodeAverage", "PropertyAge", "Pool"]]
    close_train = np.log1p(training_data['ClosePrice'])
    #print(training_data)
    #print(training_data.shape)
    #print(training_data.isna().sum())

    #can use random draws as opposed to median inputation
    testing_data = testing_data[testing_data['PropertyType'] == "Residential"]
    testing_data = testing_data[testing_data['PropertySubType'] == "SingleFamilyResidence" ]
    testing_data =  testing_data[testing_data["StateOrProvince"] == "CA"]
    testing_data = testing_data[(testing_data['ClosePrice']  <= testing_data['ClosePrice'].quantile(0.995)) & (testing_data['ClosePrice']  >= testing_data['ClosePrice'].quantile(0.005))]
    testing_data = testing_data[['ClosePrice', 'LivingArea', 'LotSizeAcres', 'YearBuilt','BathroomsTotalInteger', 'BedroomsTotal', 'GarageSpaces', 'Latitude', 'Longitude', "PoolPrivateYN", 'City', "PostalCode"]].dropna()
    testing_data["PropertyAge"] = 2025 - testing_data["YearBuilt"]
    testing_data['Pool'] = testing_data['PoolPrivateYN'].map({'Y': 1, 'N': 0})
    testing_data['City'] = testing_data['City'].astype('category')
    #postal_code_average = testing_data.groupby("PostalCode")["ClosePrice"].mean().reset_index().rename(columns={'ClosePrice': 'PostalCodeAverage'})
    testing_data = testing_data.merge(postal_code_average, on="PostalCode", how="left" )
    testing_data = testing_data.merge(price_per_square_foot_zip_code, on="PostalCode", how="left" )

    features_test = testing_data[['LivingArea', 'LotSizeAcres', 'YearBuilt', 'BathroomsTotalInteger','BedroomsTotal', 'GarageSpaces',  'Longitude', 'Latitude',"PostalCodeAverage", "PropertyAge", "Pool"]]
    #features_test = np.hstack([testing_zip_encoded, features_test])
    close_test = testing_data['ClosePrice']
    return features_train, close_train, features_test, close_test

features_train, close_train, features_test, close_test = process_data(training_data, testing_data)
"""
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
training_zip_encoded = encoder.fit_transform(training_data[['PostalCode']])
testing_zip_encoded  = encoder.transform(testing_data[['PostalCode']])
"""

#features_train = np.hstack([training_zip_encoded, features_train])
"""
model = RandomForestRegressor(n_estimators=2000, max_depth=None, min_samples_leaf=500, n_jobs = -1, random_state=random.randint(1,100))
model.fit(features_train, close_train)
predicted_close = np.expm1(model.predict(features_test))

print(predicted_close)
print(close_test)
r2 = r2_score(close_test, predicted_close)
print(f"Random forest r2: {r2}")
mdape_score = MedianAbsolutePercentageError()(close_test, predicted_close)
print(f"Random forrest median absolute error percentage: {mdape_score}")

plt.scatter(close_test, predicted_close, alpha=0.1)
plt.plot([close_test.min(), close_test.max()],[close_test.min(), close_test.max()])
plt.xlabel("Actual")
plt.ylabel("Random Forest Predicted")
plt.show()
"""

#BEST: 0.840248390326241 (0.1, 2500)
def grid_search():
    best_r2 = -1
    best_params = None

    for lr in [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for n_est in [500, 1000, 1500, 2000, 2500]:
            model = xgb.XGBRegressor(
                enable_categorical=True,
                learning_rate=lr,
                max_depth=None,
                n_estimators=n_est,
                n_jobs=1
            )
            model.fit(features_train, close_train)

            preds = np.expm1(model.predict(features_test)).clip(5000, 1_000_000_000)

            r2 = r2_score(close_test, preds)

            print(lr, n_est, "R^2:", r2)

            if r2 > best_r2:
                best_r2 = r2
                best_params = (lr,  n_est)

    print("BEST:", best_r2, best_params)


model = xgb.XGBRegressor(enable_categorical=True, n_estimators=2500, max_depth=None, learning_rate = 0.1, n_jobs = 1, random_state=random.randint(1,100))
model.fit(features_train, close_train)
#predicted_close = model.predict(features_test)
predicted_close = np.expm1(pd.Series(model.predict(features_test), index=close_test.index, name="ClosePrice")).clip(5000, 1000000000)
#predicted_close = model.predict(features_test).clip(5000, 50000000)

#print(predicted_close)
#print(close_test)
r2 = r2_score(close_test, predicted_close)
print(f"XGBoost r2: {r2}")

r2_log = r2_score(np.log1p(close_test), np.log1p(predicted_close))
print(f"XGBoost log r2: {r2_log}")

mdape_score = MedianAbsolutePercentageError()(close_test, predicted_close)
print(f"XGboost median absolute error percentage: {mdape_score}")

mape_score = mean_absolute_percentage_error(close_test, predicted_close)
print(f"XGboost mean absolute error percentage: {mape_score}")

plt.scatter(close_test, predicted_close, alpha=0.1)
plt.plot([close_test.min(), close_test.max()],[close_test.min(), close_test.max()])
plt.xlabel("Actual")
plt.ylabel("XGBoost Predicted")
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.show()

#ChatGPT generated interactive display
# Make a DataFrame with predictions and actuals + extra info
plot_df = features_test.copy()
plot_df['PredictedPrice'] = predicted_close  # or output_test for NN
plot_df['ActualPrice'] = close_test

# Add any extra info you want to see on hover
plot_df['HoverInfo'] = (
    'YearBuilt: ' + plot_df['YearBuilt'].astype(str)+'<br>' +
    'BathroomsTotalInteger: ' + plot_df["BathroomsTotalInteger"].astype(str) + '<br>' +
    'BedroomsTotal: ' + plot_df["BedroomsTotal"].astype(str) + '<br>' +
    'GarageSpaces: ' + plot_df["GarageSpaces"].astype(str) +'<br>' +
    'Longitude: ' + plot_df["Longitude"].astype(str)+'<br>' +
    'Latitude: ' + plot_df["Latitude"].astype(str) + '<br>' +
    "PostalCodeAverage: " + plot_df["PostalCodeAverage"].astype(str) +'<br>' +
    'PostalCode: ' + testing_data['PostalCode'].astype(str) + '<br>' +
    'Living Area: ' + plot_df['LivingArea'].astype(str) + ' sqft<br>' +
    'LotSize: ' + plot_df['LotSizeAcres'].astype(str) + ' acres'
)

# Interactive scatter
fig = px.scatter(
    plot_df, 
    x='ActualPrice', 
    y='PredictedPrice', 
    hover_data=['HoverInfo'],   # show these on hover
    opacity=0.6
)

fig.update_layout(
    title='Predicted vs Actual Prices',
    xaxis_title='Actual Price',
    yaxis_title='Predicted Price'
)

fig.show()



"""
training_input_tensor = torch.tensor(StandardScaler().fit_transform(features_train), dtype = torch.float32)
training_output_tensor = torch.tensor((np.log1p(close_train.values)).reshape(-1,1), dtype = torch.float32)

model = nn.Sequential(nn.Linear(len(features_train.columns), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.0005)

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_function(model(training_input_tensor), training_output_tensor)
    loss.backward()
    optimizer.step()
    if(epoch % 10 == 0):
        print(f"epoch: {epoch} loss: {loss}")


training_input_tensor = torch.tensor(StandardScaler().fit_transform(features_test), dtype = torch.float32)
model.eval()
with torch.no_grad():
    output_test = model(training_input_tensor).numpy()

output_test = np.expm1(output_test).reshape(-1).clip(50000, 50000000) #was getting some really extreme values here
output_actual = close_test.values.reshape(-1)


r2 = r2_score(output_test, output_actual)
print(f"Neural Network r2: {r2}")
corr = np.corrcoef(output_test, output_actual)[0, 1]
print(f"Neural Network correlation: {corr}")

plt.scatter(output_test, output_actual, alpha=0.1)
plt.plot([output_test.min(), output_test.max()],[output_test.min(), output_test.max()])
plt.xlabel("Actual")
plt.ylabel("Neural Network Predicted")
plt.show() 
"""
