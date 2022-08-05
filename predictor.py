### Custom definitions and classes if any ###
import joblib
import pandas as pd
import numpy as np
from preprocess import main
from trainer import train_models
def v_match(v_name):
    v_dict={'RANCHI':['JHARKHAND','RANCHI','JSCA', 'International', 'Stadium', 'Complex','Stadium','International','Complex','Cricket','Association'],
               'INTL'   :['Sheikh', 'Zayed','Sharjah','Dubai','Stadium','International','Complex','Cricket','Association'],
               'MUMBAI':['Wankhede','Brabourne','Mumbai','Stadium','International','Complex','Cricket','Association'],
               'DELHI':['Feroz', 'Shah', 'Kotla','Arun', 'Jaitley','Delhi','Green','Park','Stadium','International','Complex','Cricket','Association'],
               'BANGALORE':['BANGALORE','Chinnaswamy','M','Stadium','International','Complex','Cricket','Association'],
               'GUJRAT':['GUJRAT','Motera','Sardar', 'Patel', 'Narendra','Modi','Ahmedabad','Saurashtra','Stadium','International','Complex','Cricket','Association'],
               'HYDERABAD':['HYDERABAD','Rajiv','Gandhi','Uppal','Stadium','International','Complex','Cricket','Association'],
               'CUTTACK':['CUTTACK','Barabati','Stadium','International','Complex','Cricket','Association'],
               'KOLKATA':['KOLKATA','Eden', 'Gardens','Stadium','International','Complex','Cricket','Association'],
               'PUNJAB':['Punjab','Mohali','I','S','IS','Bindra','Stadium','International','Complex','Cricket','Association'],
               'PUNE':['PUNE','Maharashtra','Stadium','International','Complex','Cricket','Association'],
               'VIJAG':['VIJAG','Dr', 'Y','S','YS', 'Rajasekhara', 'Reddy', 'ACA-VDCA','Stadium','International','Complex','Cricket','Association'],
               'RAIPUR':['RAIPUR','Shaheed', 'Veer', 'Narayan', 'Singh','Stadium','International','Complex','Cricket','Association'],
               'INDORE':['INDORE','HOLKAR','Stadium','International','Complex','Cricket','Association'],
               'JAIPUR':['JAIPUR','Sawai','Mansingh','(SMS)','Rajasthan','Stadium','International','Complex','Cricket','Association'],
               'CHENNAI':['CHENNAI','Chepauk','M', 'A','MA','Chidambaram','Stadium','International','Complex','Cricket','Association']}  
    for key,values in v_dict.items():
        v_dict[key] = [item.strip().upper() for item in values]

    v_name = v_name.replace(',',' ').replace('.',' ').replace('  ',' ')
    v_name_l = [item.strip().upper() for item in v_name.split(' ')]

    for key,values in v_dict.items():
        check =  all(item in values for item in v_name_l)
        if check:
            return key
    return "ALL"



def predict_data(input_test):
    global in_df
    #print(encoded_columns)
    #print(len(encoded_columns))
    #read and prepare match input data
    encoded_columns = ['match_id', 'start_date', 'innings', 'total_run', 'total_w', 'total_b', 
                       'batting_team_Chennai Super Kings', 'batting_team_Delhi Capitals', 
                       'batting_team_Kings XI Punjab', 'batting_team_Kolkata Knight Riders', 
                       'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals', 
                       'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad', 
                       'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Capitals', 
                       'bowling_team_Kings XI Punjab', 'bowling_team_Kolkata Knight Riders', 
                       'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals', 
                       'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 
                       'venue_BANGALORE', 'venue_CHENNAI', 'venue_DELHI', 'venue_HYDERABAD', 'venue_INDORE', 
                       'venue_INTL', 'venue_JAIPUR', 'venue_KOLKATA', 'venue_MUMBAI', 'venue_PUNE', 
                       'venue_PUNJAB', 'venue_RAIPUR', 'venue_VIJAG', 'venue_ALL']
    inputFile_df = pd.read_csv(input_test)
    ip_data_dict = inputFile_df.iloc[0].to_dict()
    total_w = len(ip_data_dict['batsmen'].split(','))-2
    total_b = len(ip_data_dict['bowlers'].split(','))
    
    inputFile_df.loc[inputFile_df.index[0], 'venue'] = v_match(inputFile_df.loc[inputFile_df.index[0], 'venue'])
    #print(inputFile_df.loc[inputFile_df.index[0], 'venue'])
    inputFile_df.loc[inputFile_df['batting_team'].str.upper()=='Punjab Kings'.upper(),'batting_team'] = 'Kings XI Punjab'
    inputFile_df.loc[inputFile_df['bowling_team'].str.upper()=='Punjab Kings'.upper(),'bowling_team'] = 'Kings XI Punjab'
    
    inputFile_df =  pd.get_dummies(data=inputFile_df, columns=['batting_team', 'bowling_team', 'venue'])
    
    df = pd.DataFrame(columns=encoded_columns)
    inputFile_df = pd.concat([df, inputFile_df], axis=0, ignore_index=True,sort=False)
    inputFile_df.drop( columns=['batsmen','bowlers'], inplace=True)
    inputFile_df = inputFile_df[encoded_columns]
    inputFile_df.drop( columns=['match_id','start_date'], inplace=True)
    inputFile_df.fillna(value=0, axis=None, inplace=True)
    inputFile_df['total_w'] = total_w
    inputFile_df['total_b'] = total_b
    inputFile_df.to_csv('inputFile_preprocessed.csv',index=False)
    #print(inputFile_df)



def predictRuns(input_test):
    predict_data(input_test)
    main(input_test)
    train_models()

    rf_pred = joblib.load('rf_m.pkl')
    svm_rbf_pred = joblib.load('svr_rbf_m.pkl')
    svm_lin_pred = joblib.load('svr_lin_m.pkl')
    svm_poly_pred = joblib.load('svr_poly_m.pkl')
    
    df_input = pd.read_csv('inputFile_preprocessed.csv')
    df_input = df_input.drop('total_run', axis = 1)
    input_features = np.array(df_input)
    
    
    predictions_rf = rf_pred.predict(input_features)
    predictions_svm_rbf = svm_rbf_pred.predict(input_features)
    predictions_svm_lin = svm_lin_pred.predict(input_features)
    predictions_svm_poly = svm_poly_pred.predict(input_features)
    
    prediction = int((predictions_rf+predictions_svm_rbf*2+predictions_svm_lin+predictions_svm_poly)/5)
    #print(predictions_rf,predictions_svm_rbf,predictions_svm_lin,predictions_svm_poly)
    ### Your Code Here ###
    return prediction
