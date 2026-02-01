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

df202506 = pd.read_csv("raw/california/CRMLSSold202506.csv", encoding = "ISO-8859-1")
df202507 = pd.read_csv("raw/california/CRMLSSold202507.csv", encoding = "ISO-8859-1")
df202508 = pd.read_csv("raw/california/CRMLSSold202508.csv", encoding = "ISO-8859-1")
df202509 = pd.read_csv("raw/california/CRMLSSold202509.csv", encoding = "ISO-8859-1")
df202510 = pd.read_csv("raw/california/CRMLSSold202510.csv", encoding = "ISO-8859-1")
df202511 = pd.read_csv("raw/california/CRMLSSold202511.csv", encoding = "ISO-8859-1")
training_data = pd.concat([df202506, df202507, df202508, df202509, df202510, df202511])
testing_data = pd.read_csv("raw/california/CRMLSSold202512.csv", encoding = "ISO-8859-1")

print(((training_data.isna().mean() * 100).sort_values(ascending=False)))

training_data = training_data[training_data['PropertyType'] == "Residential"]
training_data = training_data[training_data['PropertySubType'] == "SingleFamilyResidence" ]
training_data = training_data[(training_data['ClosePrice']  <= training_data['ClosePrice'].quantile(0.995)) & (training_data['ClosePrice']  >= training_data['ClosePrice'].quantile(0.005))]
training_data = training_data[['ClosePrice', 'LivingArea', 'LotSizeAcres', 'YearBuilt','BathroomsTotalInteger', 'BedroomsTotal', 'GarageSpaces']].dropna()

features_train = training_data[['LivingArea', 'LotSizeAcres', 'YearBuilt', 'BathroomsTotalInteger','BedroomsTotal', 'GarageSpaces']]
close_train = training_data['ClosePrice']
print(training_data)
print(training_data.shape)
print(training_data.isna().sum())


testing_data = testing_data[testing_data['PropertyType'] == "Residential"]
testing_data = testing_data[testing_data['PropertySubType'] == "SingleFamilyResidence" ]
testing_data = testing_data[(testing_data['ClosePrice']  <= testing_data['ClosePrice'].quantile(0.995)) & (testing_data['ClosePrice']  >= testing_data['ClosePrice'].quantile(0.005))]
testing_data = testing_data[['ClosePrice', 'LivingArea', 'LotSizeAcres', 'YearBuilt','BathroomsTotalInteger', 'BedroomsTotal', 'GarageSpaces']].dropna()

features_test = testing_data[['LivingArea', 'LotSizeAcres', 'YearBuilt', 'BathroomsTotalInteger','BedroomsTotal', 'GarageSpaces']]
close_test = testing_data['ClosePrice']

"""
model = RandomForestRegressor(n_estimators=2000, max_depth=None, min_samples_leaf=500, n_jobs = -1, random_state=random.randint(1,100))
model.fit(features_train, close_train)
predicted_close = model.predict(features_test)

print(predicted_close)
print(close_test)
r2 = r2_score(close_test, predicted_close)
print(f"Random forest r2: {r2}")
corr = np.corrcoef(close_test, predicted_close)[0, 1]
print(f"Random forest correlation: {corr}")

plt.scatter(close_test, predicted_close, alpha=0.1)
plt.plot([close_test.min(), close_test.max()],[close_test.min(), close_test.max()])
plt.xlabel("Actual")
plt.ylabel("Random Forest Predicted")
plt.show()

model = xgb.XGBRegressor(n_estimators=2000, max_depth=None, learning_rate = 0.05, n_jobs = -1, random_state=random.randint(1,100))
model.fit(features_train, close_train)
predicted_close = model.predict(features_test)

print(predicted_close)
print(close_test)
r2 = r2_score(close_test, predicted_close)
print(f"XGBoost r2: {r2}")
corr = np.corrcoef(close_test, predicted_close)[0, 1]
print(f"XGboost correlation: {corr}")

plt.scatter(close_test, predicted_close, alpha=0.1)
plt.plot([close_test.min(), close_test.max()],[close_test.min(), close_test.max()])
plt.xlabel("Actual")
plt.ylabel("XGBoost Predicted")
plt.show()
"""

#
training_input_tensor = torch.tensor(StandardScaler().fit_transform(features_train), dtype = torch.float32)
training_output_tensor = torch.tensor((np.log1p(close_train.values)).reshape(-1,1), dtype = torch.float32)

model = nn.Sequential(nn.Linear(len(features_train.columns), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

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

