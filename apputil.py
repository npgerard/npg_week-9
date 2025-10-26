import pandas as pd
import numpy as np
import streamlit as st
from streamlit_gsheets import GSheetsConnection


class GroupEstimate:
    def __init__(self, estimate):
        self.estimate = estimate
        # Initialize group statistics object to be filled by fit() 
        # and referenced by predict()
        self.group_stats = None


    def fit(self, X, y):
        #Combine the datasests
        df_combined = pd.concat([X.dropna(), y.dropna()], axis=1)

        grouping_columns = X.columns.tolist()

        if self.estimate == 'mean':
            self.group_stats = df_combined.groupby(grouping_columns).mean()
        elif self.estimate == 'median':
            self.group_stats = df_combined.groupby(grouping_columns).median()

        self.group_stats.to_csv('group_stats.csv')

    def predict(self, X_):
        ''' performs predictions on X_ based on group statistics'''

        # Ensure X_ is treated as a DataFrame rather than a list
        X_ = pd.DataFrame(X_, columns=self.group_stats.index.names)

        # define the column we want to return predictions for
        prediction_column = self.group_stats.columns[0]

        # Generate predictions based on group statistics
        predictions = []
        for _, row in X_.iterrows():
            group = tuple(row[name] for name in self.group_stats.index.names)
            if group in self.group_stats.index:
                pred = self.group_stats.loc[group, prediction_column]
            else:
                pred = np.nan  # or some default value
            predictions.append(pred)
        return predictions


# def GroupEstimate(object):
#     def __init__(self, estimate):
#         self.estimate = estimate
    
#     def fit(self, X, y):
#         return None

#     def predict(self, X):
#         return None
    
# get the dataset to work with
# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

# Caches data ... setting ttl=0 will disable caching
df_coffee = conn.read(ttl="5m")


# set upt eh group_estimate object and fit for the mean
grpEstimate = GroupEstimate('mean')

# fit the model 
grpEstimate.fit(
    # X
    df_coffee[['roaster','roast']],
    # y
    df_coffee[['rating']]
    )

# ask for some predictions with the following data
X_ = [['A.R.C.', 'Medium-Light'],
      ['PT\'s Coffee Roasting', 'Light'],
      ['BC Coffee Roasters', 'Light']
    ]

# need to set up the output predictions
estimates = grpEstimate.predict(X_)

print(estimates)