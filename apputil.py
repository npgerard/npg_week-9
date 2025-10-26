import pandas as pd
import numpy as np


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
    

df_coffee = pd.read_csv('coffee_analysis.csv')

grpEstimate = GroupEstimate('mean')
grpEstimate.fit(
    # X
    df_coffee[['roaster','roast']],
    # y
    df_coffee[['rating']]
    )

X_ = [['A.R.C.', 'Medium-Light'],
      ['PT\'s Coffee Roasting', 'Light'],
      ['BC Coffee Roasters', 'Light']
    ]

estimates = grpEstimate.predict(X_)

print(estimates)