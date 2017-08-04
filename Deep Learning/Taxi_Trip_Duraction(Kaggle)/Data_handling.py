import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from six.moves import cPickle as pickle

train_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\temprary_data_analysis\\test_processed.csv')
train_data = train_data.reset_index(drop=True)

### Data Augmentation
# print('Augmenting data...')
# print(train_data.shape)
#
# def getDistance(lon1,lat1,lon2,lat2):
#     # in kilometers
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371
#     return c * r
# def getSpeed(index):
#     distance = getDistance(train_data.at[index,'pickup_longitude'],train_data.at[index,'pickup_latitude'],
#                                                          train_data.at[index,'dropoff_longitude'],train_data.at[index,'dropoff_latitude'])
#     time = train_data.at[index,'trip_duration']/3600
#     return distance/time
#
# train_data = train_data.assign(trip_distance=0.0)
# for i in range(train_data.shape[0]):
#     train_data.set_value(i,'trip_distance',getDistance(train_data.at[i,'pickup_longitude'],train_data.at[i,'pickup_latitude'],
#                                                          train_data.at[i,'dropoff_longitude'],train_data.at[i,'dropoff_latitude']))
#
# target = None
# if target != None and input('Save temp data?') == 'Y':
#     file_name = input('Enter file name: ')
#     with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\temprary_data_analysis\\{}'.format(file_name),'wb') as f:
#         save = {'target':target}
#         pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
#
#
# print(train_data.shape)
#
# if input('PROCEED?') != 'Y':
#     assert False
# train_data.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\temprary_data_analysis\\test_processed.csv',index=False,header=True)
# assert False
###

# print('Retreiving weather data...')
# weather_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\weather_data_nyc_centralpark_2016.csv')
# weather_dict = {}
# for i in range(weather_data.shape[0]):
#     month = int(re.findall('\d+', weather_data.at[i, 'date'])[1])
#     day = int(re.findall('\d+', weather_data.at[i, 'date'])[0])
#     if month in weather_dict:
#         weather_dict[month][day] = [weather_data.at[i,'precipitation'],weather_data.at[i,'snow fall'],
#                                     weather_data.at[i,'snow depth'],weather_data.at[i,'average temperature']/100,
#                                     weather_data.at[i,'maximum temperature']/100,weather_data.at[i,'minimum temperature']/100]
#     else:
#         weather_dict[month] = {day:[weather_data.at[i,'precipitation'],weather_data.at[i,'snow fall'],
#                                     weather_data.at[i,'snow depth'],weather_data.at[i,'average temperature']/100,
#                                weather_data.at[i, 'maximum temperature'], weather_data.at[i, 'minimum temperature']]}

print('Modifying time data...')
train_data = train_data.assign(normalized_pickup_time = 0.0, week_day = 0)
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
    train_data.set_value(i, 'normalized_pickup_time', ((float(pickup_data_time[3]) * 60 + float(pickup_data_time[4])) / 1440.0)-0.5)
    train_data.set_value(i, 'week_day', weekDay(int(pickup_data_time[0]),int(pickup_data_time[1]),int(pickup_data_time[2]))+1)

# print('Modifying trip duration...')
# train_data = train_data.assign(time_duration = 0.0) ##
# for i in range(train_data.shape[0]): ##
#     train_data.set_value(i,'time_duration',float(train_data.at[i,'trip_duration'])/60.0) ##

# print('Adding weather data...')
# train_data = train_data.assign(precipitation=0.00,snow_fall=0.00,snow_depth=0.00,average_temperature=0.00,maximum_temperature=0.00,minimum_temperature=0.00)
# for i in range(train_data.shape[0]):
#     month = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[1])
#     day = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[2])
#     train_data.set_value(i,'precipitation',weather_dict[month][day][0])
#     train_data.set_value(i, 'snow_fall', weather_dict[month][day][1])
#     train_data.set_value(i, 'snow_depth', weather_dict[month][day][2])
#     train_data.set_value(i, 'average_temperature', weather_dict[month][day][3])
#     train_data.set_value(i, 'maximum_temperature', weather_dict[month][day][4])
#     train_data.set_value(i, 'minimum_temperature', weather_dict[month][day][5])

print("Selecting features...")
# train_data = train_data[['passenger_count','normalized_pickup_time','trip_distance','week_day','time_duration']] ##
train_data = train_data[['id','passenger_count','normalized_pickup_time','trip_distance','week_day']] #


print("Separating weekday and weekend...")
train_dataset_1 = train_data[(train_data['week_day'] != 6) & (train_data['week_day'] != 7)]
train_dataset_2 = train_data[(train_data['week_day'] == 6) | (train_data['week_day'] == 7)]

del train_dataset_1['week_day']
del train_dataset_2['week_day']

# train_labels_1 = train_dataset_1[['time_duration']] ##
# train_labels_2 = train_dataset_2[['time_duration']] ##
# del train_dataset_1['time_duration'] ##
# del train_dataset_2['time_duration'] ##

# train_dataset_1 = train_dataset_1.reset_index(drop=True) ##
# train_dataset_2 = train_dataset_2.reset_index(drop=True) ##
# train_labels_1 = train_labels_1.reset_index(drop=True) ##
# train_labels_2 = train_labels_2.reset_index(drop=True) ##

print('Finalizing data structure...')
# train_dataset_1 = train_dataset_1.loc[:,:].as_matrix() ##
# train_dataset_2 = train_dataset_2.loc[:,:].as_matrix() ##
#
# train_labels_1 = train_labels_1.loc[:,:].as_matrix() ##
# train_labels_2 = train_labels_2.loc[:,:].as_matrix() ##

test_dataset_1 = train_dataset_1.loc[:,:].as_matrix() #
test_dataset_2 = train_dataset_2.loc[:,:].as_matrix() #

print(test_dataset_1.shape) #
print(test_dataset_2.shape) #

# print(train_dataset_1.shape) ##
# print(train_dataset_2.shape) ##
# print(train_labels_1.shape) ##
# print(train_labels_2.shape) ##


if input('proceed?') != 'Y':
    assert False

# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_1.pickle','wb') as f: ##
#     save = {'train_dataset':train_dataset_1,'train_labels':train_labels_1} ##
#     pickle.dump(save,f,protocol=2) ##
#
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','wb') as f: ##
#     save = {'train_dataset':train_dataset_2,'train_labels':train_labels_2} ##
#     pickle.dump(save, f, protocol=2) ##

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_1.pickle','wb') as f: #
    save = {'test_dataset':test_dataset_1} #
    pickle.dump(save, f, protocol=2) #

with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_2.pickle','wb') as f: #
    save = {'test_dataset':test_dataset_2} #
    pickle.dump(save, f, protocol=2) #

####################################
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_1.pickle','rb') as f:
#     save = pickle.load(f)
#     train_dataset = save['train_dataset']
#     train_labels = save['train_labels']
#     length = train_dataset.shape[0]
#
# valid_dataset = train_dataset[:80000]
# valid_labels = train_labels[:80000]
# train_dataset = train_dataset[80000:]
# train_labels = train_labels[80000:]
# print(train_dataset.shape)
# print(train_labels.shape)
# print(valid_dataset.shape)
# print(valid_labels.shape)
#
# if input('proceed?') != 'Y':
#     assert False
#
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_1.pickle','wb') as f:
#     save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels}
#     pickle.dump(save,f,protocol=2)
#
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','rb') as f:
#     save = pickle.load(f)
#     train_dataset = save['train_dataset']
#     train_labels = save['train_labels']
#
# valid_dataset = train_dataset[:40000]
# valid_labels = train_labels[:40000]
# train_dataset = train_dataset[40000:]
# train_labels = train_labels[40000:]
# print(train_dataset.shape)
# print(train_labels.shape)
# print(valid_dataset.shape)
# print(valid_labels.shape)
#
# if input('proceed?') != 'Y':
#     assert False
#
# with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_2.pickle','wb') as f:
#     save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels}
#     pickle.dump(save,f,protocol=2)
