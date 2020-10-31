#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Siddhartha Jagannath
"""

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from prediction.covid_utils import CovidUtils


class PovertyLevelTransformer(BaseEstimator, TransformerMixin):
    

    
    for year in file_dict.keys():
                
            df = pd.read_csv(file_dict[year],
                             skiprows=2,
                             usecols=lambda col_name: col_name not in ['Total',
                                                                       'Footnotes', '100%+']
        
            df = df.rename({
                            'At or Below 99%' : 'percent_below_FPL',
                            }, axis=1)    
            
            # Remove all rows below Wyoming; they are notes
            wyoming_idx = df.loc[df['State'] == 'Wyoming'].index[0]
            df = df[df.index <= wyoming_idx]
            
            abbrevs = self.utils.state_abbrev_series(df.State)
            
            indx = pd.MultiIndex.from_tuples(zip(abbrevs, 
                                                 [year]*len(abbrevs)
                                                 ), 
                                                 names=['Region','Election'])
            df.index = indx
            # No longer need State column, b/c we have
            # that info in the index:
            df = df.drop('Location', axis=1)
            
            if self.all_elections_df is None:
                self.all_elections_df = df
            else:
                self.all_elections_df = pd.concat([self.all_elections_df, df])

    #------------------------------------
    # fit
    #-------------------

    def fit(self, X):
        return self
    
    #------------------------------------
    # transform 
    #-------------------

    def transform(self, X):
        
        new_X = X.join(self.all_elections_df)
        return new_X