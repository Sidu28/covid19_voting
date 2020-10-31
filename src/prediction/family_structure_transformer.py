#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:37:23 2020

@author: sidu
"""

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from prediction.covid_utils import CovidUtils


class familyStructureTransformer(BaseEstimator, TransformerMixin):
    
    '''
    Data source: https://www.kff.org/other/state-indicator/distribution-by-family-structure/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D
    '''
    
    
    for year in file_dict.keys():
                
            df = pd.read_csv(file_dict[year],
                             skiprows=2,
                             usecols=lambda col_name: col_name not in ['Total',
                                                                       'Footnotes']
        
            df = df.rename({'Location' : 'State',
                            'Asian/Native Hawaiian and Pacific Islander' : 'Asian_Hawaiin_Pacific',
                            'American Indian/Alaska Native' : 'Native_American',
                            'Multiple Races' : 'Multiracial'}, axis=1)    
            
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
            df = df.drop('State', axis=1)
            
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
        '''
        Adjoin the age distribution for each
        state to the passed-in df. 
        Assumptions: 
           o len(X) is number of States plus D.C, but w/o Puerto Rico
           o X.index are tuples (<State_abbrev>, <year>)
           
        Returns a copy of X with age distribution in the following
        new columns:
            Age_19_25
            Age_26_34
            Age_35_54
            Age_55_64
            Age_65_up
           
        @param X: feature matrix 
        @type X: pd.DataFrame
        @return: copy of X with new columns appended
        @rtype: pd.DataFrame
        '''
        
        new_X = X.join(self.all_elections_df)
        return new_X