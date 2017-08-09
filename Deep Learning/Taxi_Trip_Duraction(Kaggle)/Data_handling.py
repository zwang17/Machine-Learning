import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from six.moves import cPickle as pickle

train_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final.csv')
train_data = train_data.reset_index(drop=True)

##### Data Processing
# print('Augmenting data...')
# print(train_data.shape)
# original_len = train_data.shape[0]
# def getDistanceE(lon1,lat1,lon2,lat2):
#     # in kilometers
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371
#     return c * r
# def getDistanceM(lon1,lat1,lon2,lat2):
#     return getDistanceE(lon1,lat1,lon2,lat1)+getDistanceE(lon2,lat2,lon2,lat1)
#
# def getSpeed(index):
#     distance = train_data.at[index,'trip_distance_Manhattan']
#     time = train_data.at[index,'trip_duration']/3600
#     return distance/time
#
# def weekDay(year, month, day):
#     offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
#     afterFeb = 1
#     if month > 2: afterFeb = 0
#     aux = year - 1700 - afterFeb
#     dayOfWeek  = 5
#     dayOfWeek += (aux + afterFeb) * 365
#     dayOfWeek += aux // 4 - aux // 100 + (aux + 100) // 400
#     dayOfWeek += offset[month - 1] + (day - 1)
#     dayOfWeek %= 7
#     return dayOfWeek
#
# print('Calculating Euclidean trip distance...')
# train_data = train_data.assign(trip_distance_Euclidean=0.0)
# for i in range(train_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/original_len*100,'%')
#     train_data.set_value(i,'trip_distance_Euclidean',getDistanceE(train_data.at[i,'pickup_longitude'],train_data.at[i,'pickup_latitude'],
#                                                          train_data.at[i,'dropoff_longitude'],train_data.at[i,'dropoff_latitude']))
# print('Calculating Manhattan trip distance...')
# train_data = train_data.assign(trip_distance_Manhattan=0.0)
# for i in range(train_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/train_data.shape[0]*100,'%')
#     train_data.set_value(i,'trip_distance_Manhattan',getDistanceM(train_data.at[i,'pickup_longitude'],train_data.at[i,'pickup_latitude'],
#                                                           train_data.at[i,'dropoff_longitude'],train_data.at[i,'dropoff_latitude']))
# # print('Calculating speed...')
# # train_data = train_data.assign(speed=0.0)
# # for i in range(train_data.shape[0]):
# #     if i % 5000 == 0:
# #         print(i/original_len*100,'%')
# #     train_data.set_value(i,'speed',getSpeed(i))
# # print('Dropping extreme values...')
# # train_data = train_data[train_data['speed']<140]
# # train_data = train_data[train_data['speed']>0.9]
# # print('Dropped count: ',original_len - train_data.shape[0])
# # train_data = train_data.reset_index(drop=True)
#
# print('Normalizing passenger count...')
# train_data['normalized_passenger_count'] = train_data['passenger_count']
# train_data['normalized_passenger_count'] = train_data['passenger_count'].apply(lambda x: str(x))
# for i in range(train_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/train_data.shape[0]*100,'%')
#     train_data.set_value(i,'normalized_passenger_count',train_data.at[i,'passenger_count']/10.0)
#
# print('Modifying time data...')
# train_data = train_data.assign(normalized_pickup_time = 0.0, week_day = 0)
# for i in range(train_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/train_data.shape[0]*100,'%')
#     pickup_data_time = re.findall('\d+', train_data.at[i, 'pickup_datetime'])
#     train_data.set_value(i, 'normalized_pickup_time', ((float(pickup_data_time[3]) * 60 + float(pickup_data_time[4])) / 1440.0)-0.5)
#     train_data.set_value(i, 'week_day', weekDay(int(pickup_data_time[0]),int(pickup_data_time[1]),int(pickup_data_time[2])))
#
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
# print('Adding weather data...')
# train_data = train_data.assign(precipitation=0.00,snow_fall=0.00,average_temperature=0.00,maximum_temperature=0.00,minimum_temperature=0.00)
# for i in range(train_data.shape[0]):
#     month = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[1])
#     day = int(re.findall('\d+', train_data.at[i,'pickup_datetime'])[2])
#     train_data.set_value(i,'precipitation',weather_dict[month][day][0])
#     train_data.set_value(i, 'snow_fall', weather_dict[month][day][1])
#     train_data.set_value(i, 'average_temperature', weather_dict[month][day][3])
#     train_data.set_value(i, 'maximum_temperature', weather_dict[month][day][4])
#     train_data.set_value(i, 'minimum_temperature', weather_dict[month][day][5])
#
# print(train_data.shape)
# print(train_data[:30])
#
# while input('PROCEED?') != 'Y':
#     print('Invalid input')
# train_data.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final.csv',index=False,header=True)
# train_data[:50].to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final_sample.csv',index=False,header=True)
# assert False
####################################################################
##### Data Augmentation
# pickup_longitude_sdv = 0.04
# pickup_latitude_sdv = 0.03
# dropoff_longitude_sdv = 0.04
# dropoff_latitude_sdv = 0.03
# trip_distance_Euclidean_sdv = 4
# trip_distance_Manhattan_sdv = 5.5
# normalized_passenger_count_sdv = 0.13
# normalized_pickup_time_sdv = 0.27
# average_temperature_sdv = 0.15
# maximum_temperature_sdv = 0.17
# minimum_temperature_sdv =0.15
# precipitation_sdv = 0.3
# snow_fall_sdv = 0.9
# train_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_processed_final.csv')
# train_data = train_data.reset_index(drop=True)
# for i in range(train_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/train_data.shape[0]*100,'%')
#     train_data.set_value(i,'normalized_pickup_longitude',  train_data.at[i,'pickup_longitude']/pickup_longitude_sdv)
#     train_data.set_value(i, 'normalized_pickup_latitude', train_data.at[i, 'pickup_latitude'] / pickup_latitude_sdv)
#     train_data.set_value(i, 'normalized_dropoff_longitude', train_data.at[i, 'dropoff_longitude'] / dropoff_longitude_sdv)
#     train_data.set_value(i, 'normalized_dropoff_latitude', train_data.at[i, 'dropoff_latitude'] / dropoff_latitude_sdv)
#     train_data.set_value(i, 'normalized_trip_distance_Euclidean', train_data.at[i, 'trip_distance_Euclidean'] / trip_distance_Euclidean_sdv)
#     train_data.set_value(i, 'normalized_trip_distance_Manhattan', train_data.at[i, 'trip_distance_Manhattan'] / trip_distance_Manhattan_sdv)
#     train_data.set_value(i, 'normalized_passenger_count', train_data.at[i, 'normalized_passenger_count'] / normalized_passenger_count_sdv)
#     train_data.set_value(i, 'normalized_pickup_time', train_data.at[i, 'normalized_pickup_time'] / normalized_pickup_time_sdv)
#     train_data.set_value(i, 'normalized_average_temperature', train_data.at[i, 'average_temperature'] / average_temperature_sdv)
#     train_data.set_value(i, 'normalized_maximum_temperature', train_data.at[i, 'maximum_temperature'] / maximum_temperature_sdv)
#     train_data.set_value(i, 'normalized_minimum_temperature', train_data.at[i, 'minimum_temperature'] / minimum_temperature_sdv)
#     train_data.set_value(i, 'normalized_precipitation', train_data.at[i, 'precipitation'] / precipitation_sdv)
#     train_data.set_value(i, 'normalized_snow_fall', train_data.at[i, 'snow_fall'] / snow_fall_sdv)
# while input('PROCEED?') != 'Y':
#     print('Invalid input')
# train_data.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_processed_final.csv',index=False,header=True)
# train_data[:50].to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_processed_final_sample.csv',index=False,header=True)
#
# test_data = pd.read_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final.csv')
# test_data = test_data.reset_index(drop=True)
# for i in range(test_data.shape[0]):
#     if i % 5000 == 0:
#         print(i/test_data.shape[0]*100,'%')
#     test_data.set_value(i,'normalized_pickup_longitude',  test_data.at[i,'pickup_longitude']/pickup_longitude_sdv)
#     test_data.set_value(i, 'normalized_pickup_latitude', test_data.at[i, 'pickup_latitude'] / pickup_latitude_sdv)
#     test_data.set_value(i, 'normalized_dropoff_longitude', test_data.at[i, 'dropoff_longitude'] / dropoff_longitude_sdv)
#     test_data.set_value(i, 'normalized_dropoff_latitude', test_data.at[i, 'dropoff_latitude'] / dropoff_latitude_sdv)
#     test_data.set_value(i, 'normalized_trip_distance_Euclidean', test_data.at[i, 'trip_distance_Euclidean'] / trip_distance_Euclidean_sdv)
#     test_data.set_value(i, 'normalized_trip_distance_Manhattan', test_data.at[i, 'trip_distance_Manhattan'] / trip_distance_Manhattan_sdv)
#     test_data.set_value(i, 'normalized_passenger_count', test_data.at[i, 'normalized_passenger_count'] / normalized_passenger_count_sdv)
#     test_data.set_value(i, 'normalized_pickup_time', test_data.at[i, 'normalized_pickup_time'] / normalized_pickup_time_sdv)
#     test_data.set_value(i, 'normalized_average_temperature', test_data.at[i, 'average_temperature'] / average_temperature_sdv)
#     test_data.set_value(i, 'normalized_maximum_temperature', test_data.at[i, 'maximum_temperature'] / maximum_temperature_sdv)
#     test_data.set_value(i, 'normalized_minimum_temperature', test_data.at[i, 'minimum_temperature'] / minimum_temperature_sdv)
#     test_data.set_value(i, 'normalized_precipitation', test_data.at[i, 'precipitation'] / precipitation_sdv)
#     test_data.set_value(i, 'normalized_snow_fall', test_data.at[i, 'snow_fall'] / snow_fall_sdv)
# while input('PROCEED?') != 'Y':
#     print('Invalid input')
# test_data.to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final.csv',index=False,header=True)
# test_data[:50].to_csv('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_processed_final_sample.csv',index=False,header=True)
# assert False
####################################################################
##### Pickling
# train_switch = False
#
# print("Selecting features...")
# # train_data = train_data[['normalized_passenger_count','normalized_pickup_time','normalized_trip_distance_Euclidean',
# #                          'normalized_trip_distance_Manhattan','normalized_pickup_longitude','normalized_pickup_latitude',
# #                          'normalized_dropoff_longitude','normalized_dropoff_latitude','normalized_average_temperature',
# #                          'normalized_maximum_temperature','normalized_minimum_temperature','normalized_precipitation',
# #                          'normalized_snow_fall','vendor_id','week_day','trip_duration']] ##
# train_data = train_data[['id','normalized_passenger_count','normalized_pickup_time','normalized_trip_distance_Euclidean',
#                          'normalized_trip_distance_Manhattan','normalized_pickup_longitude','normalized_pickup_latitude',
#                          'normalized_dropoff_longitude','normalized_dropoff_latitude','normalized_average_temperature',
#                          'normalized_maximum_temperature','normalized_minimum_temperature','normalized_precipitation',
#                          'normalized_snow_fall','vendor_id','week_day']] #
#
#
# print("Separating weekdays and vendors...")
# train_dataset_1_0 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 0)]
# train_dataset_1_1 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 1)]
# train_dataset_1_2 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 2)]
# train_dataset_1_3 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 3)]
# train_dataset_1_4 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 4)]
# train_dataset_1_5 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 5)]
# train_dataset_1_6 = train_data[(train_data['vendor_id'] == 1) & (train_data['week_day'] == 6)]
# train_dataset_2_0 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 0)]
# train_dataset_2_1 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 1)]
# train_dataset_2_2 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 2)]
# train_dataset_2_3 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 3)]
# train_dataset_2_4 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 4)]
# train_dataset_2_5 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 5)]
# train_dataset_2_6 = train_data[(train_data['vendor_id'] == 2) & (train_data['week_day'] == 6)]
#
# train_data_dic = {'train_dataset_1_0':train_dataset_1_0,'train_dataset_1_1':train_dataset_1_1,'train_dataset_1_2':train_dataset_1_2,
#                   'train_dataset_1_3':train_dataset_1_3,'train_dataset_1_4':train_dataset_1_4,'train_dataset_1_5':train_dataset_1_5,
#                   'train_dataset_1_6':train_dataset_1_6,'train_dataset_2_0':train_dataset_2_0,'train_dataset_2_1':train_dataset_2_1,
#                   'train_dataset_2_2':train_dataset_2_2,'train_dataset_2_3':train_dataset_2_3,'train_dataset_2_4':train_dataset_2_4,
#                   'train_dataset_2_5':train_dataset_2_5,'train_dataset_2_6':train_dataset_2_6}
# train_labels_dic = {}
#
# if train_switch == True:
#     for v in [1,2]:
#         for i in range(7):
#             train_labels_dic['train_labels_{}_{}'.format(v,i)] = train_data_dic['train_dataset_{}_{}'.format(v,i)][['trip_duration']]
# for v in [1,2]:
#     for i in range(7):
#         del train_data_dic['train_dataset_{}_{}'.format(v,i)]['week_day'],\
#             train_data_dic['train_dataset_{}_{}'.format(v,i)]['vendor_id']
#         if train_switch == True:
#             del train_data_dic['train_dataset_{}_{}'.format(v,i)]['trip_duration']
#
# for v in [1,2]:
#     for i in range(7):
#         train_data_dic['train_dataset_{}_{}'.format(v, i)] = train_data_dic['train_dataset_{}_{}'.format(v,i)].reset_index(drop=True)
#         if train_switch == True:
#             train_labels_dic['train_labels_{}_{}'.format(v, i)] = train_labels_dic['train_labels_{}_{}'.format(v,i)].reset_index(drop=True)
#
# print('Finalizing data structure...')
# for v in [1,2]:
#     for i in range(7):
#         train_data_dic['train_dataset_{}_{}'.format(v, i)] = train_data_dic['train_dataset_{}_{}'.format(v, i)].loc[:,:].as_matrix()
#         if train_switch == True:
#             train_labels_dic['train_labels_{}_{}'.format(v, i)] = train_labels_dic['train_labels_{}_{}'.format(v, i)].loc[:,:].as_matrix()
#
# for v in [1,2]:
#     for i in range(7):
#         print(train_data_dic['train_dataset_{}_{}'.format(v, i)].shape)
#         if train_switch == True:
#             print(train_labels_dic['train_labels_{}_{}'.format(v, i)].shape)
#
# while input('proceed?') != 'Y':
#     print('Invalid input')
#
# for v in [1,2]:
#     for i in range(7):
#         if train_switch == True:
#             with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_{}_{}.pickle'.format(v,i),'wb') as f:
#                 save = {'train_dataset':train_data_dic['train_dataset_{}_{}'.format(v, i)]}
#                 save['train_labels'] = train_labels_dic['train_labels_{}_{}'.format(v,i)]
#                 pickle.dump(save,f,protocol=2)
#         else:
#             with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\test_data_final\\test_{}_{}.pickle'.format(v,i),'wb') as f:
#                 save = {'test_dataset':train_data_dic['train_dataset_{}_{}'.format(v, i)]}
#                 pickle.dump(save,f,protocol=2)
# assert False
####################################
##### Data Partitioning
for v in [1,2]:
    for i in range(7):
        with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_{}_{}.pickle'.format(v,i),'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            length = train_dataset.shape[0]
        print(length)
        valid_size = int(input('Validation size: '))
        valid_dataset = train_dataset[:valid_size]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:]
        train_labels = train_labels[valid_size:]
        print(train_dataset.shape)
        print(train_labels.shape)
        print(valid_dataset.shape)
        print(valid_labels.shape)

        while input('proceed?') != 'Y':
            print('Invalid input')

        with open('D:\\Google Drive\\Deep_Learning_Data\Data\Taxi Trip Duration(Kaggle)\\train_data_final\\train_{}_{}.pickle'.format(v,i),'wb') as f:
            save = {'train_dataset':train_dataset,'train_labels':train_labels,'valid_dataset':valid_dataset,'valid_labels':valid_labels}
            pickle.dump(save,f,protocol=2)
