import pandas as pd
import numpy as np

if __name__ == "__main__":

    # reading the train and test data
    train_data = pd.read_csv('power_gen/input/train_folds.csv')
    test_data = pd.read_csv('power_gen/input/test.csv')

    # concatinating train and test data for cleaning data
    merged_data = pd.concat([train_data, test_data], axis=0)

    # replacing filler values into missing values
    merged_data = merged_data.replace(-99,np.nan)
    merged_data = merged_data.replace(-999,np.nan)
    merged_data = merged_data.replace(999,np.nan)

    # taking absolute values of some features
    merged_data['wind_speed(m/s)'] = merged_data['wind_speed(m/s)'].abs()
    merged_data['atmospheric_pressure(Pascal)'] = merged_data['atmospheric_pressure(Pascal)'].abs()
    merged_data['blade_length(m)'] = merged_data['blade_length(m)'].abs()
    merged_data['windmill_height(m)'] = merged_data['windmill_height(m)'].abs()

    # converting angle values between 0-360 degrees
    # subtracting 360 from the values grater than 360
    for index, row in merged_data.iterrows():
        if row['wind_direction(°)'] == np.nan:
            pass
        elif row['wind_direction(°)'] >= 360:
            merged_data['wind_direction(°)'][index] = row['wind_direction(°)'] - 360

    # adding 360 to the negative values
    for index, row in merged_data.iterrows():
        if row['blades_angle(°)'] == np.nan:
            pass
        elif row['blades_angle(°)']<0:
            merged_data['blades_angle(°)'][index] = row['blades_angle(°)'] + 360

    # extracting features from datetime column
    merged_data['datetime'] = pd.to_datetime(merged_data['datetime'], format="%Y-%m-%d %H:%M:%S")
    merged_data['year']= merged_data['datetime'].dt.year
    merged_data['month']= merged_data['datetime'].dt.month
    merged_data['day']= merged_data['datetime'].dt.day
    merged_data['hour']= merged_data['datetime'].dt.hour
    merged_data['minute']= merged_data['datetime'].dt.minute
    merged_data['second']= merged_data['datetime'].dt.second

    merged_data['current_date'] = pd.Timestamp('2020-12-31')
    date_data1 = merged_data['current_date']
    date_data2 = merged_data['datetime']
    merged_data['days_diff'] = (date_data1-date_data2).dt.days

    # Seperating train and test data
    train_data_mod = merged_data.iloc[:28200, :]
    test_data_mod = merged_data.iloc[28200:, :].drop(columns = ['windmill_generated_power(kW/h)', 'kfold'], axis = 1)

    # filling missing values
    num_missing_cols = ['wind_speed(m/s)', 'atmospheric_temperature(°C)', 'shaft_temperature(°C)', 'blades_angle(°)', 'gearbox_temperature(°C)',
                        'engine_temperature(°C)', 'motor_torque(N-m)', 'generator_temperature(°C)', 'atmospheric_pressure(Pascal)',
                    'windmill_body_temperature(°C)',	'wind_direction(°)',	'resistance(ohm)',	'rotor_torque(N-m)',
                    'blade_length(m)',	'windmill_height(m)', 'area_temperature(°C)', 'windmill_generated_power(kW/h)']
    num_missing_cols_t = num_missing_cols[:-1]
    cat_missing_cols = ['turbine_status',	'cloud_level']

    # for col in num_missing_cols:
    #     train_data_mod[col].fillna(train_data_mod[col].mean(), inplace=True)

    # for col in num_missing_cols_t:
    #     test_data_mod[col].fillna(train_data_mod[col].mean(), inplace=True)

    # for col in cat_missing_cols:
    #     train_data_mod[col].fillna(train_data_mod[col].mode()[0], inplace=True)
    #     test_data_mod[col].fillna(train_data_mod[col].mode()[0], inplace=True)

    # # one hot encoding categorical features
    # ohe_cat_cols = ['turbine_status', 'cloud_level']
    # one_hot_data = train_data_mod[ohe_cat_cols]
    # one_hot_data = pd.get_dummies(one_hot_data, drop_first=True, prefix=ohe_cat_cols)
    # one_hot_data_t = test_data_mod[ohe_cat_cols]
    # one_hot_data_t= pd.get_dummies(one_hot_data_t, drop_first=True, prefix=ohe_cat_cols)

    # train_data_mod.drop(ohe_cat_cols, inplace = True, axis=1)
    # test_data_mod.drop(ohe_cat_cols, inplace = True, axis=1)
    # cooked_data = pd.concat([train_data_mod, one_hot_data],axis = 1)
    # cooked_data_t = pd.concat([test_data_mod, one_hot_data_t],axis = 1)

    cooked_data = train_data_mod
    cooked_data_t = test_data_mod

    # dropping unimportant features
    cooked_data.drop(['tracking_id', 'datetime', 'current_date'], inplace = True, axis=1)
    base_sub = cooked_data_t[['tracking_id', 'datetime']]
    cooked_data_t.drop(['tracking_id', 'datetime', 'current_date'], inplace = True, axis=1)

    cooked_data.to_csv("power_gen/input/train_processed_xg.csv", index=False)
    cooked_data_t.to_csv("power_gen/input/test_processed_xg.csv", index=False)