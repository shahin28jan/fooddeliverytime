import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            '''#Calculate distance by Haversine formula (used to find the distance between two geographical locations)

            R = 6371  ##The earth's radius (in km)

            def deg_to_rad(degrees):
                return degrees * (np.pi/180)

            ## The haversine formula
                def distcalculate(lat1, lon1, lat2, lon2):
                    d_lat = deg_to_rad(lat2-lat1)
                    d_lon = deg_to_rad(lon2-lon1)
                    a1 = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1))
                    a2 = np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
                    a = a1 * a2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    return R * c

            # Create distance column & calculate the distance
                df['distance'] = np.nan

                for i in range(len(df)):
                    df.loc[i, 'distance'] = distcalculate(df.loc[i, 'Restaurant_latitude'], 
                    df.loc[i, 'Restaurant_longitude'], 
                    df.loc[i, 'Delivery_location_latitude'], 
                    df.loc[i, 'Delivery_location_longitude'])

            ## In given data many columns are not affect the delivery time, so I drop those columns

            df = df.drop(labels=['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked', 'Restaurant_latitude',	'Restaurant_longitude',	'Delivery_location_latitude', 'Delivery_location_longitud', 'Type_of_order'],axis=1)


            ## Independent and dependent feature
            X = df.drop(labels=['Time_taken (min)'],axis=1)
            Y = df[['Time_taken (min)']]'''


            # Define which columns should be ordinal-encoded and which should be scaled

            categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle','Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Vehicle_condition','multiple_deliveries']

            # Define the custom ranking for each ordinal variable
            Weather_condition_categories = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            Road_traffic_density_categories = ['Jam', 'High', 'Medium', 'Low']
            Type_of_vehicle_categories = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            Festival_categories = ['No', 'Yes']
            City_categories = ['Metropolitian', 'Urban', 'Semi-Urban']
                      
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())

            ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
            steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder',OrdinalEncoder(categories=[Weather_condition_categories,Road_traffic_density_categories, Type_of_vehicle_categories, Festival_categories, City_categories])),
            ('scaler',StandardScaler())
            ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name, 'ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Type_of_order', 'Time_Order_picked']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
                