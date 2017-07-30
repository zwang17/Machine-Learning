import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from six.moves import cPickle as pickle

train_data = pd.read_csv('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train.csv')

del train_data['vendor_id'], train_data['store_and_fwd_flag']

train_data = train_data.reset_index(drop=True)

print('Retreiving weather data...')
weather_data = pd.read_csv('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\weather_data_nyc_centralpark_2016.csv')
weather_dict = {}
for i in range(weather_data.shape[0]):
    month = int(re.findall('\d+', weather_data.at[i, 'date'])[1])
    day = int(re.findall('\d+', weather_data.at[i, 'date'])[0])
    if month in weather_dict:
        weather_dict[month][day] = [weather_data.at[i,'precipitation'],weather_data.at[i,'snow fall'],
                                    weather_data.at[i,'snow depth'],weather_data.at[i,'average temperature']/100,
                                    weather_data.at[i,'maximum temperature']/100,weather_data.at[i,'minimum temperature']/100]
    else:
        weather_dict[month] = {day:[weather_data.at[i,'precipitation'],weather_data.at[i,'snow fall'],
                                    weather_data.at[i,'snow depth'],weather_data.at[i,'average temperature']/100,
                               weather_data.at[i, 'maximum temperature'], weather_data.at[i, 'minimum temperature']]}
print('Adding weather data...')
train_data = train_data.assign(precipitation=0.00,snow_fall=0.00,snow_depth=0.00,average_temperature=0.00,maximum_temperature=0.00,minimum_temperature=0.00)
for i in range(train_data.shape[0]):
    month = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[1])
    day = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[2])
    train_data.set_value(i,'precipitation',weather_dict[month][day][0])
    train_data.set_value(i, 'snow_fall', weather_dict[month][day][1])
    train_data.set_value(i, 'snow_depth', weather_dict[month][day][2])
    train_data.set_value(i, 'average_temperature', weather_dict[month][day][3])
    train_data.set_value(i, 'maximum_temperature', weather_dict[month][day][4])
    train_data.set_value(i, 'minimum_temperature', weather_dict[month][day][5])

train_data = train_data.assign(normalized_pickup_time = 0.0, trip_distance = 0.0, week_day = 0)

print('Modifying time data...')
def weekDay(year, month, day):
    offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    afterFeb = 1
    if month > 2: afterFeb = 0
    aux = year - 1700 - afterFeb
    dayOfWeek  = 5
    dayOfWeek += (aux + afterFeb) * 365
    dayOfWeek += aux // 4 - aux // 100 + (aux + 100) // 400
    dayOfWeek += offset[month - 1] + (day - 1)
    dayOfWeek %= 7
    return dayOfWeek

for i in range(train_data.shape[0]):
    pickup_data_time = re.findall('\d+', train_data.at[i, 'pickup_datetime'])
    train_data.set_value(i, 'normalized_pickup_time', (float(pickup_data_time[3]) * 60 + float(pickup_data_time[4])) / 1440.0)
    train_data.set_value(i, 'week_day', weekDay(int(pickup_data_time[0]),int(pickup_data_time[1]),int(pickup_data_time[2]))+1)

del train_data['pickup_datetime'],train_data['dropoff_datetime'] ##

print('Calculating trip distance...')
def getDistance(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r
for i in range(train_data.shape[0]):
    train_data.set_value(i,'trip_distance',getDistance(train_data.at[i,'pickup_longitude'],train_data.at[i,'pickup_latitude'],
                                                         train_data.at[i,'dropoff_longitude'],train_data.at[i,'dropoff_latitude']))

del train_data['pickup_longitude'],train_data['pickup_latitude'],train_data['dropoff_longitude'],train_data['dropoff_latitude']

print('Calculating time duration...')
# test_dataset = train_data[['id','passenger_count','normalized_pickup_time','trip_distance','precipitation','snow_depth','snow_fall','average_temperature','week_day']] #
train_dataset = train_data[['passenger_count','normalized_pickup_time','trip_distance','precipitation','snow_depth','snow_fall','average_temperature','maximum_temperature','minimum_temperature']] ##
train_labels = train_data[['trip_duration']] ##

train_data = train_data.assign(time_duration = 0.0) ##

for i in range(train_labels.shape[0]): ##
    train_labels.set_value(i,'time_duration',float(train_labels.at[i,'trip_duration'])/60.0) ##

del train_labels['trip_duration'] ##


print('Finalizing data structure...')
train_dataset = train_dataset.loc[:,:].as_matrix() ##
train_labels = train_labels.loc[:,:].as_matrix() ##


# test_dataset = test_dataset_1.loc[:,:].as_matrix() #

# print(test_dataset.shape) #
print(train_dataset.shape) ##
print(train_labels.shape) ##

print(train_dataset) ##
print(train_labels) ##

if input('proceed?') != 'Y':
    assert False

with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train.pickle','wb') as f:
    save = {'train_dataset':train_dataset,'train_labels':train_labels}
    pickle.dump(save,f,protocol=2)

# with open('C:\\Users\\zheye1218\\Google Drive\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train.pickle','rb') as f:
#     save = pickle.load(f)
#     print(save)
#     train_dataset = save['train_dataset']
#     train_labels = save['train_labels']
#
# print(train_dataset.shape)
# print(train_labels.shape)
