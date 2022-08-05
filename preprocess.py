# -*- coding: utf-8 -*-
"""Preprocess"""
### Custom definitions and classes if any ###
import pandas as pd



in_df = pd.DataFrame()
encoded_columns=[]

def read_data():
    global in_df
    input_file = 'all_matches.csv'
    req_columns = ['match_id','start_date','venue','innings','ball','bowler','wicket_type','batting_team','bowling_team',
                   'runs_off_bat','extras']
    in_df = pd.read_csv(input_file,usecols = req_columns)
    
def filter_data():  
    global in_df,encoded_columns,v_match 
    #apply filter for date>=2014 and over<6.0
    #in_df['start_date'] = in_df['start_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    in_df['start_date'] = pd.to_datetime(in_df['start_date'], format='%Y-%m-%d')
    in_df = in_df[in_df['start_date'].dt.year >= 2016]
    in_df = in_df[in_df['ball'] < 6.0]
    in_df = in_df[in_df['innings'] < 3]
    
    #replace NaN with zero
    in_df[['extras', 'runs_off_bat']] = in_df[['extras','runs_off_bat']].fillna(value=0)
    
    #add a new column by adding runs of bat and extras
    in_df['total_run'] = in_df['extras']+in_df['runs_off_bat']
    
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

    #makes team names and venues consistent. In future, only these names will be allowed to read from the input data
    in_df['venue'] = in_df['venue'].apply(lambda x: v_match(x))
    in_df = in_df.replace(to_replace ='Delhi Daredevils',value = 'Delhi Capitals')
    
    #drops tems that are not a part of the current IPL season
    consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad']
    in_df = in_df[(in_df['batting_team'].isin(consistent_teams)) & (in_df['bowling_team'].isin(consistent_teams))]
    
    #get total score after 6 overs
    in_df['total_w']=0
    in_df['total_b'] = 0
    in_df['wicket_type'] = in_df['wicket_type'].apply(lambda x: 1 if not pd.isnull(x) else 0)
    in_df['total_run'] = in_df.groupby(['match_id','innings'])['total_run'].transform('sum')
    in_df['total_w'] = in_df.groupby(['match_id','innings'])['wicket_type'].transform('sum')
    in_df['total_b'] = in_df.groupby(['match_id','innings'])['bowler'].transform('nunique')

    #print(in_df['venue'].unique())
    op_df = in_df.drop_duplicates(subset=['match_id','innings'], keep='first')
    op_df = op_df[['match_id','start_date','venue','innings','batting_team','bowling_team','total_run','total_w','total_b']]
    encoded_df = pd.get_dummies(data=op_df, columns=['batting_team', 'bowling_team', 'venue'])
    #print(encoded_df.head)
    encoded_df['venue_ALL'] = 1
    #encoded_columns = list(encoded_df.columns)
    encoded_df.to_csv('all_matches_preprocessed.csv',index=False)
    

def main(input_test):
    read_data()
    filter_data()
    #predict_data(input_test)

main('inputFile.csv')
