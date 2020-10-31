'''
Created on Sep 4, 2020

@author: paepcke
'''

import os, sys
import pickle

from category_encoders.binary import BinaryEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from matplotlib import rcParams
import openpyxl  # for Excel exports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from population_age_transformer import PopulationAgeTransformer
from poverty_by_race_transformer import PovertybyRaceTransformer
from prediction.covid_utils import CovidUtils
from utils.logging_service import LoggingService
from visualization import Visualizer


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))





class StatePredictor(object):
    '''
    classdocs
    '''
    VOTER_TURNOUT_FILES = {
                           2004: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2004NovemberGeneralElection.xlsx'),
                           2006: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2006NovemberGeneralElection.xlsx'),
                           2008: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2008NovemberGeneralElection.xlsx'),
                           2010: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2010NovemberGeneralElection.xlsx'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/turnoutRates2012NovemberGeneralElection.xlsx'),
                           2014: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2014NovemberGeneralElection.xlsx'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/turnoutRates2016NovemberGeneralElection.xlsx'),
                           2018: os.path.join(os.path.dirname(__file__),
                                       '../../data/turnoutRates2018NovemberGeneralElection.xlsx'),
                           }
    
    QUERY_TERM_FILES    = {
                           (2004,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2004_vote.csv'),      # 'vote'
                           (2006,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2006_vote.csv'),
                           (2008,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2008_vote.csv'),
                           (2010,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2010_vote.csv'),
                           (2012,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2012_vote.csv'),
                           (2014,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2014_vote.csv'),
                           (2016,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2016_vote.csv'),
                           (2018,'vote'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2018_vote.csv'),
                           (2004,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2004_elections.csv'),  # 'elections'
                           (2006,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2006_elections.csv'),
                           (2008,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2008_elections.csv'),
                           (2010,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2010_elections.csv'),
                           (2012,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2012_elections.csv'),
                           (2014,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2014_elections.csv'),
                           (2016,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2016_elections.csv'),
                           (2018,'elections'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2018_elections.csv'),
                           (2004,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2004_voting.csv'),     # 'voting'
                           (2006,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2006_voting.csv'),
                           (2008,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2008_voting.csv'),
#********
#                            (2010,'voting'): os.path.join(os.path.dirname(__file__),
#                                             '../../data/dataset_2010_voting.csv'),
#********
                           (2012,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2012_voting.csv'),
                           (2014,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2014_voting.csv'),
                           (2016,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2016_voting.csv'),
                           (2018,'voting'): os.path.join(os.path.dirname(__file__),
                                            '../../data/dataset_2018_voting.csv'),

                          }
    VOTER_TURNOUT_DEMOGRAPHICS = os.path.join(os.path.dirname(__file__),
                                            '../../data/turnoutByRaceAgeEduCPS.xlsx')
    
    RACE_ETHNICITY_BY_STATE = {
                           2008: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2008.csv'),
                           2010: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2010.csv'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2012.csv'),
                           2014: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2014.csv'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2016.csv'),
                           2018: os.path.join(os.path.dirname(__file__),
                                            '../../data/raceEthnicityByState2018.csv')
                           }
    
    AGE_BY_STATE =         {
                           2008: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2008.csv'),
                           2010: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2010.csv'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2012.csv'),
                           2014: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2014.csv'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2016.csv'),
                           2018: os.path.join(os.path.dirname(__file__),
                                            '../../data/age2018.csv')
                           }

    MAIL_VOTING =          {
                            2018: os.path.join(os.path.dirname(__file__),
                                               '../../voteByMail2018.xlsx')
                            2016: os.path.join(os.path.dirname(__file__),
                                               '../../voteByMail2016.xls'),
                            2014: os.path.join(os.path.dirname(__file__),
                                               '../../voteByMail2014.xls'),
            
                            
                            2012: os.path.join(os.path.dirname(__file__),
                                               '../../voteByMail2012.xlsx'),
                           }
    POVERTY_BY_RACE_BY_STATE =         {
                           2008: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2008.csv'),
                           2010: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2010.csv'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2012.csv'),
                           2014: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2014.csv'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2016.csv'),
                           2018: os.path.join(os.path.dirname(__file__),
                                            '../../data/povertyrace2018.csv')
                           }
    '''
    FAMILY_STRUCTURE_BY_STATE =         {
                           2008: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2008.csv'),
                           2010: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2010.csv'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2012.csv'),
                           2014: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2014.csv'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2016.csv'),
                           2018: os.path.join(os.path.dirname(__file__),
                                            '../../data/structure2018.csv')
                           }
    '''
    FPL_BY_STATE =         {
                           2008: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2008.csv'),
                           2010: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2010.csv'),
                           2012: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2012.csv'),
                           2014: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2014.csv'),
                           2016: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2016.csv'),
                           2018: os.path.join(os.path.dirname(__file__),
                                            '../../data/fpl2018.csv')
                           }

    RANDOM_SEED = 42

    #------------------------------------
    # Constructor 
    #-------------------
    

    def __init__(self, label_col='VoterTurnout'):
        '''
        Constructor
        '''
        self.log = LoggingService()
        self.utils = CovidUtils()
        
        # Get charts to layout properly (e.g. show
        # all of the axis labels:
        rcParams.update({'figure.autolayout': True})


        # Directory with various data files:
        self.script_dir = os.path.dirname(__file__) 
        self.data_dir   = os.path.join(self.script_dir, '../../data')

        # Initialize various mappings (State names to their abbrevs, etc.),
        self.utils.import_state_mappings()
        
        # Import voter turnout:
        self.log.info("Importing voter turnout data (Election Project)...")
        voter_turnout = self.import_voter_turnout(self.VOTER_TURNOUT_FILES)
        self.log.info("Done importing voter turnout data (Election Project).")

        # Import demographics of voter turnout:
        self.log.info("Importing voting-eligible populations by race for each State...")
        voter_turnout = self.add_turnout_demographics(voter_turnout)
        self.log.info("Done importing voting-eligible populations by race for each State.")        

        # Import csv file with Google query statistics:
        self.log.info("Importing Google search keyword counts (Google Trends)...")
        search_features = self.import_search_data(self.QUERY_TERM_FILES)
        self.log.info("Done importing Google search keyword counts (Google Trends).")

        # Join voter turnout and search frequencies 
        # into one wide table:
        self.log.info("Joining Google search with voter turnout...")
        election_features = self.merge_turnout_query_counts(voter_turnout, 
                                                            search_features)
        self.log.info("Done joining Google search with voter turnout.")
        
        # Resort the 2-tier index to keep performance, but
        # mostly to avoid the warning about performance degraded
        # due to unsorted multiindex:
        
        election_features = election_features.sort_index()
        
        # Add information about election time disasters:
        # Whether or not (col 'Disaster', and the name
        # of the disaster (col 'Disaster_Name'):

        self.log.info("Adding disaster history...")
        election_features = self.add_disaster_information(election_features)
        self.log.info("Done adding disaster history.")
        
        self.log.info("Adding age distribution by State...")
        election_features = PopulationAgeTransformer(self.AGE_BY_STATE).fit_transform(election_features)
        self.log.info("Done adding age distribution by State.")
        
        self.log.info("Adding Poverty  by race distribution...")
        election_features = PovertybyRaceTransformer(self.POVERTY_BY_RACE_BY_STATE.fit_transform(election_features)
        self.log.info("Done Poverty  by race distribution")
        
        self.log.info("Adding FPL distribution by state...")
        election_features = PovertyLevelTransformer(self.FPL_BY_STATE.fit_transform(election_features)
        self.log.info("Done Adding FPL distribution by state...")

'''
        self.log.info("Population Distribution by Family Structure by State...")
        election_features = FamilyStructureTransformer(self.FAMILY_STRUCTURE_BY_STATE).fit_transform(election_features)
        self.log.info("Done Adding Population Distribution by Family Structure by State")
'''
        # Remove columns not needed, and encode 
        # categorical columns:
        
        # StateCode starts with consecutive ints
        # Must encode those:
        y = election_features[label_col]
        new_state_codes = self.leave_one_out_encode(cat_col=election_features['StateCode'],
                                                    label_col=y)
        election_features['StateCode'] = new_state_codes
        
        # Remove the ASCII State and disaster names:
        election_features = election_features.drop(columns=['State'])
        election_features = election_features.drop(columns=['DisasterName'])
        
        # Encode the Google query terms:
        queries_encoded = self.binary_encode(election_features['Query'])
        election_features = pd.concat([election_features, queries_encoded], axis=1)
        election_features = election_features.drop('Query', axis=1)
        
        # Since weekdays do have some interval characteristics,
        # just encode them as 0,1,...6:
        week_day_encoded = self.ordinal_encode(election_features['WeekDay'])
        election_features['WeekDay'] = week_day_encoded

        # Get (again, just in case) the values in column that we are to predict:
        y = election_features[label_col]
        
        # Remove that col from election_features.
        # Also, some features are meaningless, or closely
        # related to some other feature. Remove
        # those:

        X = election_features.drop(columns=[label_col,
                                            'StateCode',
                                            'TotalBallotsCounted'
                                            ],
                                            axis=1)
        
        # Save the feature names (without the label col),
        # and the multiindex before having to turn
        # X into an np array. Same with other
        # elements in the dataframe structure:
        self.election_years = pd.Series(X.index.get_level_values(1).unique(),
                                        dtype=int)
        self.feature_names = X.columns
        self.feature_index = X.index
        self.target_name   = label_col
        self.X_df          = X.copy()
        self.y_series      = y.copy()
        
        # Make the final feature vectors (incl. label_col)
        # available to other methods:
        self.election_features = election_features
        
        #*****
#         election_features.to_pickle(os.path.join(os.path.dirname(__file__),
#                                        '../../data/SavedFrames/features.pickle')
#                                     )
#         election_features.to_excel(os.path.join(os.path.dirname(__file__),
#                                        '../../data/SavedFrames/features.xlsx')
#                                     )
        #*****
        
        #**********
        #self.correlation_matrix(X,y)
        #**********
        self.log.info("Creating RandomForestRegressor...")
        self.rand_forest = RandomForestRegressor()

        self.log.info("Done creating RandomForestRegressor.")

    #------------------------------------
    # run
    #-------------------

    def run(self):
        
        for election_yr in self.election_years:
            
            # For a given election, we can only use
            # feature values from the past, or from 
            # measurements taken just before the election,
            # like query counts. But not from the future:
            
            past_only_features = self.X_df.query(f"Election <= {election_yr}")
            
            # For Series, query() does not work. Use masks instead:
            
            mask = self.y_series.index.get_level_values('Election') <= election_yr
            past_only_target   = self.y_series[mask]

            (pred_series_unique, 
             truth_series_unique) = self.predict_one_election(past_only_features, 
                                                              past_only_target) 
        
        self.evaluate_model(pred_series_unique, truth_series_unique)
        
    #------------------------------------
    # predict_one_election 
    #-------------------

    def predict_one_election(self, X_df, y_series):
        '''
        Get features X_df like:
        
                             Year  NonCitizenPerc  ...  Query_1  Query_2
            Region Election                        ...                  
            AK     2008      2008        0.037000  ...        0        1
                   2008      2008        0.037000  ...        0        1
                   2008      2008        0.037000  ...        0        1
                   2008      2008        0.037000  ...        1        0
                   2008      2008        0.037000  ...        1        0
            ...               ...             ...  ...      ...      ...
            WY     2018      2018        0.024732  ...        1        1
                   2018      2018        0.024732  ...        1        1

        and true turnout values as:
        
            Region  Election
            AK      2008        0.68300
                    2008        0.68300
                    2008        0.68300
                    2008        0.68300
                    2008        0.68300
                                 ...   
            WY      2018        0.47861
                    2018        0.47861

        

        @param X_df:
        @type X_df:
        @param y_series:
        @type y_series:
        @param election_year:
        @type election_year:
        '''

        # Split the data into training and testing sets
        (self.train_features_df, 
         self.test_features_df, 
         self.train_labels_series, 
         self.test_labels_series) = train_test_split(X_df, y_series, test_size=0.25, random_state=42)

        # RandomForestClassifier/Regressor want
        # pure numpies:
        
        self.X = self.train_features_df.reset_index(drop=True).to_numpy(dtype=float)
        self.y = self.train_labels_series.reset_index(drop=True).to_numpy(dtype=float)
        self.X_test = self.test_features_df.reset_index(drop=True).to_numpy(dtype=float)
        self.y_test = self.test_labels_series.reset_index(drop=True).to_numpy(dtype=float)

        # Do we have optimal parameters from previous run?
        
        rf_optimal_parms_path = os.path.join(self.script_dir, 'best_params.pickle')
        try:
            with open(rf_optimal_parms_path, 'rb') as fd:
                best_params = pickle.load(fd)
                # The '**' signals that best_params
                # is a dict, and should be used as
                # kwargs:
                self.rand_forest.set_params(**best_params)
        except FileNotFoundError:
            # No previously stored parms
            best_params = self.optimize_hyperparameters(self.X, self.y)
            with open(rf_optimal_parms_path, 'wb') as fd:
                # Save as a text format:
                pickle.dump(best_params, fd, protocol=0)

        self.log.info("Training the regressor...")
        self.rand_forest.fit(self.X, self.y)
        self.log.info("Done training the regressor.")
        predictions = self.rand_forest.predict(self.X_test)

        # Turn predictions back into a Series:
        pred_series = pd.Series(predictions, name='VoterTurnout', 
            index=self.test_labels_series.index)
        
        # There are multiple copies of the TurnoutRate
        # for each (<State>,<year>) pair. Get the unique
        # values by grouping by 'Region' and 'Election'.
        # Each group will have identical VoterTurnout values.
        # Get get just one, take the group means. Since values
        # are identical within each group, no data are lost:
        
        gb_pred = pred_series.groupby(['Region', 'Election'])
        pred_series_unique = gb_pred.mean()
        gb_truth = self.test_labels_series.groupby(['Region', 'Election'])
        truth_series_unique = gb_truth.mean()
        return (pred_series_unique, truth_series_unique)


    #------------------------------------
    # optimize_hyperparameters 
    #-------------------
    
    def optimize_hyperparameters(self, train_X, train_y):
        '''
        Takes a feature matrix and target, and returns
        a dictionary with the best Random Forest parameters.
        Assumes self.rand_forest is an uninitialized instance
        of RandomForestClassifier.
        
        Example of return: {'max_depth': 3, 'n_estimators': 8}
        
        @param train_X: feature matrix
        @type train_X: pd.DataFrame
        @param train_y: target vector
        @type train_y: pd.Series
        '''
        
        tuned_parameters = {#****'n_estimators' : 1+np.array(range(10)),
                            'n_estimators' : 1+np.array(range(10)),
                            #****'max_depth'    : 1+np.array(range(10))
                            'max_depth'    : 1+np.array(range(3))
                            }
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        clf = GridSearchCV(
            self.rand_forest,
            tuned_parameters,
            scoring=scorer,
            cv=5,
            n_jobs=10,
            verbose=1
            )
        clf.fit(train_X, 
                train_y
                )
        self.rand_forest = clf.best_estimator_
        
        importances = self.rand_forest.feature_importances_
        # Associate the importances with their feature
        # names as tuples (<feature_name>, <importance>):
        feature_importances = list(zip(self.feature_names,
                                       importances
                                       ))
        sorted_importances = sorted(feature_importances,
                                    key=lambda name_imp_pair: name_imp_pair[1],
                                    reverse=True
                                    )
        self.log.info(f'Feature importances: \n{sorted_importances}')
        
        #print(f'Best params: {clf.best_params_}')
        #print(f'Best score: {clf.best_score_}')
        return clf.best_params_

    #------------------------------------
    # train_model 
    #-------------------
    
    def train_model(self):
        
        pass


    #------------------------------------
    # evaluate_model
    #-------------------

    def evaluate_model(self, predictions, test_labels):
        # Baseline model the mean of voter turnout:
        baseline_preds    = test_labels.mean()
        baseline_errors   = abs(baseline_preds - test_labels)
        mean_baseline_err = round(np.mean(baseline_errors), 2)

        self.log.info(f"Baseline mean abs err: {100*mean_baseline_err} percentage points.")

        rmse = round(mean_squared_error(test_labels,
                                        predictions,
                                        squared=False
                                        ),2)
        
        self.log.info(f'RMSE against across all states: {rmse}')
        
        # Get RMSE between voter turnout prediction and truth
        # from predicting voter turnout for the elections within
        # each State:
        self.log.info('Computing RMSE values per voting region...')
        rmse_values = self.rmse_statewise(predictions, test_labels)
        self.log.info('Done computing RMSE values per voting region.')

        # Check feature importance:
        viz = Visualizer()
        #*****viz.feature_importance(self.rand_forest, self.feature_names)
        # Visualize panel of states, each chart
        # showing a truth vs. prediction scatterplot:
        
        viz.real_and_estimated(predictions, 
                               test_labels,
                               rmse_values)


    #------------------------------------
    # import_search_data 
    #-------------------
    
    def import_search_data(self, search_data_dict):
        '''
        Read CSV file, and return a dataframe
        containing the feature vectors of Google
        searches for voting related terms 
        
        Assumptions:
           o First seven columns are query term counts for each
             day of the week.
           o Last column is label to predict (State in this
             case, but could be voter turnout)
             
             Example rows:
                 76,100,92,54,71,69,46,0,2008,vote     # record for week3 before election State 0
                 76,57,67,85,100,96,99,0,2008,vote     # record for week2 before election State 0
                 43,60,52,100,73,80,47,0,2008,vote     # record for week1 before election State 0
                 100,67,0,61,42,33,84,1,2008,vote      # record for week3 before election State 1
                 0,47,100,55,56,46,50,1,2008,vote      # record for week2 before election State 1
                 44,42,65,49,100,42,81,1,2008,vote     # record for week1 before election State 1
                 61,42,77,100,79,44,91,2,2008,vote                   ...       State 2

           o Consecutive rows are for weeks before election.
             Example: if data contains two weeks for each 
             label (State/voter turnout), then these rows
             represent data for two consecutive weeks before
             the election.
             
        Returns a dataframe, where '*' in the last three
        records is the sum of searches over all States on that day:
        
                State Query  Week0   Week1    Week2    Mon   Tue  Wed  Thu  Fri  Sat  Sun 
           ...    AL   vote    1       0        0       76  100   92   54   71   69   46
                  AL   vote    0       1        0       76   57   67   85  100   96   99
                  AL   vote    0       0        1       43   60   52  100   73   80   47
                  AK   vote    1       0        0      100   67    0   61   42   33   84
                    ...
                  US   vote    *   *    *    *    *    *    * 
                  US   vote    *   *    *    *    *    *    * 
                  US   vote    *   *    *    *    *    *    *

        In addition, a multiindex ('Region', 'Election') is installed,
        allowing selections such as:
        
        Ex1:   select all Wyoming data for the 2016 election:
           
            search_query_df.xs(['WY',2016])
            ==> Region Election                                                           
                WY     2016        WY  2016  vote     1   0   0  49    0    0  100   58   93   45
                       2016        WY  2016  vote     0   1   0  80    0   58   68   61  100    0
                       2016        WY  2016  vote     0   0   1  00    0    0   63    0   47   45
                       2016        WY  2016  vote     1   0   0  49    0    0  100   58   93   45
                       2016        WY  2016  vote     0   1   0  80    0   58   68   61  100    0
                       2016        WY  2016  vote     0   0   1  00    0    0   63    0   47   45
                       2016        WY  2016  vote     1   0   0  49    0    0  100   58   93   45
                       2016        WY  2016  vote     0   1   0  80    0   58   68   61  100    0
                       2016        WY  2016  vote     0   0   1  00    0    0   63    0   47   45

        Ex2.   select all vote counts of week 0
            search_query_df[search_query_df.Week == 0]
            ==> Region Election                                ...                              
                AL     2008        AL  2008  vote     1   0  0  46  ...    58    56    27    75   100
                AK     2008        AK  2008  vote     1   0  0   0  ...    81   100     0    76    74
                AZ     2008        AZ  2008  vote     1   0  0  91  ...    86    60    52    72    57
                AR     2008        AR  2008  vote     1   0  0  ...    90    96     0   100    67
                
        @param csv_file: comma separated data file 
        @type csv_file: str
        @return: tuple with matrix n x 8: one week indicator
            and seven day's of Google query search counts.
            Final column is state abbrev
        @rtype: (pd.DataFrame, pd.Series)
        '''

        search_query_df = None
        for (year, query_term) in search_data_dict.keys():
            
            csv_file = search_data_dict[(year,query_term)]
            df = pd.read_csv(csv_file,
                             names=['Mon','Tue','Wed','Thu','Fri','Sat','Sun',
                                    'StateCode', 'Query'],
                             index_col=False, # 1st col is *not* an index col
                             )
            
            # Add a column that indexes the week before
            # the election that one row represents.
            # A few steps needed:
            
            # How many weeks of data for each State?
            # Take State 0 (any will do), and could how
            # often it occurs in the State column:
            
            self.num_weeks  = len(df[df.StateCode == 0])
    
            # How many States are in the data? Find
            # by counting unique values in the States
            # column: 
            num_states = len(df.groupby('StateCode').nunique())

            # A new year col:
            df['Year'] = year

            # Create the Week col. Example for 3 weeks
            # of data:
            #     0
            # 0
            # 1
            # 2
            # 0
            # 1
            # 2
            # ... one triplet for each State:

            df['Week'] = [wk_idx for wk_idx in range(self.num_weeks)] * num_states

            # Build a row for each week that is
            # the sum of all queries on one day
            # across all States:
            #
            #    week   Mon                     Tue
            #      0  sum(of Mondays of wk0)    sum(of Tuesdays of wk0)    ...
            #      1  sum(of Mondays of wk1)    sum(of Tuesdays of wk0)    ...
            
            df_final = df.copy()
            for wk in range(self.num_weeks):
                that_wk_only = df[df['Week'] == wk]
                
                # Now sum all the weekday cols, 
                # getting a series [sum(Mon), sum(Tue),...]
                wk_summed = that_wk_only[['Mon','Tue','Wed','Thu','Fri','Sat','Sun']].sum(axis=0)
                
                # Add the StateCode, Query, and Week:
                wk_summed['StateCode']  = self.utils.reverse_state_codings['US']
                wk_summed['Query']      = that_wk_only.loc[wk,'Query']
                wk_summed['Year']       = year
                wk_summed['Week']       = wk
    
                df_final = df_final.append(wk_summed, ignore_index=True)
                
            # Move the State column into left-most position:
            #df_final = df_final.reindex(columns=['State', 'Year', 'Query', 'Week', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            # We now have columns:
            #   Index(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'StateCode', 'Query', 'Week']

            if search_query_df is None:
                search_query_df = df_final.copy()
            else:
                search_query_df = search_query_df.append(df_final.copy())

        # Turn the weekday columns into two colums:
        #  'WeekDay' and 'DayCount'
        
        search_query_df_folded = self.fold_columns(search_query_df,
                                                   ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], 
                                                   'WeekDay', 
                                                   'DayCount') 

        # We are returning the following columns:
        
        #['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'StateCode', 'Query',
        # 'Year', 'Week']
        # 
        # And this multiindex:
        #
        # MultiIndex([('AL',  2004),
        #    ('AL',  2004),
        #    ('AL',  2004),
        #        ...
        #                ('WY', 20018),
        #    ('WY', 20018),
        #    ('US', 20018),
        #    ('US', 20018),
        #    ('US', 20018)],

        # Make a Region/Year multiindex:        
        # Get from the State code to the long State name to the State abbrev:

        state_codes   = search_query_df_folded['StateCode']
        idx_state = pd.Series([self.utils.state_codings[state_code] 
                               for state_code in state_codes])
        idx_year  = search_query_df_folded['Year']
        idx_state.name = 'Region'
        idx_year.name = 'Election'
        new_idx = pd.MultiIndex.from_frame(pd.DataFrame({'Region' : idx_state.to_numpy(), 
                                                         'Election' : idx_year}
                                                         ))
        search_query_df_folded.index = new_idx
        return search_query_df_folded
    

    #------------------------------------
    # import_voter_turnout 
    #-------------------
    
    def import_voter_turnout(self, excel_file_dict):
        '''
        Import an Excel sheet of voter turnouts
        as provided for download at
        http://www.electproject.org/home/precinct_data 
        
        The Excel sheet is read 
            > df.columns 
            Index(['index', 'State', 'Website', 'Status', 'VEP Total Ballots Counted',
                   'VEP Highest Office', 'VAP Highest Office', 'Total Ballots Counted',
                   'Highest Office', 'Voting Eligible Population', 'Voting Age Population',
                   'Non Citizen Perc', 'Prison', 'Probation', 'Parole',
                   'Total Ineligible Felons', 'Overseas Eligible', 'State Abbrv'],
                  dtype='object')
                  
        @param excel_src: path to spreadshee of one elections as
            provided by the Election Project.
        @type excel_src: src
        @return: data frame with the above columns.
        @rtype: pd.DataFrame

        '''
        
        voter_turnout_df = None
        for year in excel_file_dict.keys():
            
            excel_src = excel_file_dict[year]

            # The 'header=[0,1]' notifies the read method
            # that the first two rows are taken by a nested
            # header. The result will feature a multiindex:
            df = pd.read_excel(excel_src, header=[0,1])

            # Here is magic to deal with the complex,
            # merged header columns. The solution comes from: 
            # https://stackoverflow.com/questions/42132663/fix-dataframe-columns-when-reading-an-excel-file-with-a-header-with-merged-cells
    
            df = df.reset_index()

            # Turn multiindex level names that Pandas brought in
            # as 'Unnamed: ...' to emtpy strings:
            df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')

            # Drop multi-index level 'Turnout Rates', 'Numerators, 
            # 'Denominators', 'VEP Components (Modifications to VAP to Calculate VEP) 
            # We only want the flat col header of level 1:
            df.columns = df.columns.droplevel(0)
            
            # Some files have two extra columns: 'State Results Website',
            # 'Status', 'Source', and/or 'State Abv. Others don't.
            # Remove those cols if present:
            
            try:
                df = df.drop(columns=['State Results Website'], axis=1)
            except KeyError:
                pass
            try:
                df = df.drop(columns=['Status'], axis=1)
            except KeyError:
                pass
            try:
                df = df.drop(columns=['Source'], axis=1)
            except KeyError:
                pass
            try:
                df = df.drop('State Abv', axis=1)
            except KeyError:
                pass
            
            # Overseas Eligible Voters are only available
            # for the entire US; at State level the col is
            # all nan:
            
            df = df.drop('Overseas Eligible', axis=1)
            
            # Some of the Election Project sheets have 
            # Notes and such after Wyoming. Remove those:
            
            wyoming_idx_obj = df.index[(df.iloc[:,1] == 'Wyoming') == True]
            wyoming_idx     = wyoming_idx_obj.values[0]
            df = df[df.index <= wyoming_idx]
            
            # The 2018 table from Election Project is missing
            # the VAP Highest Office column. Get those numbers
            # via the Census:
            
            if year == 2018:

                vap_highest_office = self.compute_2018_VAPHighestOffice()
                # Get location of 'VEP Highest Office', after which
                # we then place the data:
                dest = df.columns.get_loc('2018 Vote for Highest Office VEP Turnout Rate')
                df.insert(dest,'VAPHighestOffice',vap_highest_office.to_numpy())
            
            # 15 columns (each of which corresponds to a level in
            # the multiindex:
            try:
                df.columns = ['index_dup', 'State', 'VoterTurnout', 'VEPHighestOffice',
                       'VAPHighestOffice', 'TotalBallotsCounted', 'HighestOffice', 'VotingEligiblePopulation',
                       'VotingAgePopulation',
                       'NonCitizenPerc',
                       'Prison',
                       'Probation',
                       'Parole',
                       'TotalIneligibleFelons',
                       ]
            except Exception as e:
                # Without this try/except, program
                # just dies if mismatch in num of 
                # cols in df vs. the above list:
                raise e
    
            # Add a year col just after the State:
            df.insert(2,'Year',[year]*len(df)) 
            
            # Remove the index_dup col:
            df = df.drop(columns='index_dup')

            # We use 'VEP Total Ballots Counted' for voter participation. 
            # But some States don't report this number. Their value will 
            # be NaN. In that case we use the 'VEP Highest Office' percentage:
            
            df['VoterTurnout'] = df['VoterTurnout'].fillna(df['VEPHighestOffice'])
            
            # Same with 'Total Ballots Counted':
            df['TotalBallotsCounted'] = df['TotalBallotsCounted'].fillna(df['HighestOffice'])
    
            # The last row may have 'Notes' in it, delete such a row:
            #if df['State']['Unnamed: 0_level_1'].iloc[-1][-1].startswith('Note:'):
            #    us_plus_state_names = us_plus_state_names[:-1]['United States']
             
            # The State column may have row(s) that are notes.
            # We hope they'll continue to start with 'Note':
            df = df.drop(df[df.State.str.startswith('Note')].index)
            
            if voter_turnout_df is None:
                voter_turnout_df  = df.copy()
                sum_prev_turnouts = df.VoterTurnout 
                num_elections = 1
            else:
                voter_turnout_df = voter_turnout_df.append(df.copy())
                voter_turnout_df['MeanPastTurnout']  = sum_prev_turnouts / num_elections 
                sum_prev_turnouts = sum_prev_turnouts + df.VoterTurnout
                num_elections += 1

        # Set a two-element index to allow
        #   df.loc['CA', 2016] to get all 2016
        # California results. We get the data
        # for the two index levels from columns
        # 'State' and 'Year', but we rename them
        # to 'Region' and 'Election'
        
        state_abbrevs = self.utils.state_abbrev_series(voter_turnout_df.State)
        region_series         = state_abbrevs
        election_series       = voter_turnout_df['Year'].copy()
        region_series.name    = 'Region'
        election_series.name  = 'Election'
        voter_turnout_df = voter_turnout_df.set_index([region_series, election_series])

        # Make some stats available for later, then 
        # drop them from the features:
        
        self.voting_eligible_population = voter_turnout_df['VotingEligiblePopulation']
        self.voting_age_population      = voter_turnout_df['VotingAgePopulation']
        
        # Drop some columns that are unlikely features:
        voter_turnout_df = voter_turnout_df.drop([
                            'VEPHighestOffice',
                            'VAPHighestOffice',
                            'HighestOffice',
                            'VotingEligiblePopulation',
                            'VotingAgePopulation',
                            'Prison',
                            'Probation',
                            'Parole'
                            ],
                            axis=1)

        

        
        return voter_turnout_df

    #------------------------------------
    # add_turnout_demographics 
    #-------------------
    
    def add_turnout_demographics(self, voter_turnout):
        '''
        Given output of import_voter_turnout(), add columns
        with absolute numbers of voting-eligible populations:
        VEP_White, VEP_Black, VEP_Hispanic, and VEP_Other. The
        numbers are based on the corrected CPS Census survey
        http://www.electproject.org/home/voter-turnout/demographics
        
        It is fine if columns beyond what import_voter_turnout()
        produces have been added before calling this method. 
        
        @param voter_turnout: table as produced by 
            import_voter_turnout(), or wider.
        @type voter_turnout: pd.DataFrame
        @return new table with the additional columns
        @rtype pd.DataFrame
        '''
        
        excel_src = self.VOTER_TURNOUT_DEMOGRAPHICS 
        
        # The 'header=[0,1]' notifies the read method
        # that the first two rows are taken by a nested
        # header. The result will feature a multiindex:
        df = pd.read_excel(excel_src, 
                           sheet_name='Race and Ethnicity',
                           header=[0])
        # We now have:
        #    Census Weight for Vote Overreport Bias Correction\n  ...  Unnamed: 17
        # 0                                        Turnout Rate   ...     1986.000
        # 1                                  Non-Hispanic White   ...        0.398
        # 2                                  Non-Hispanic Black   ...        0.358
        # 3                                            Hispanic   ...        0.282
        # 4                                               Other   ...        0.310
        # 5                                                 NaN   ...          NaN
        # 6                                 Share of Electorate   ...     1986.000
        # 7                                  Non-Hispanic White   ...        0.850
        # 8                                  Non-Hispanic Black   ...        0.100
        # 9                                            Hispanic   ...        0.036
        # 10                                              Other   ...        0.015
        # 11                                                NaN   ...          NaN
        #                      ... Goes on with unwanted

                
        # Fashion appropriate col headers:
        df.columns = ['PopSegment'] + list(range(2018,1984,-2))
        
        # Drop left-over header line:
        df = df.drop(0)
        
        # Giving us:
        #              PopSegment       2018       2016  ...      1990      1988      1986
        # 1    Non-Hispanic White     0.5521     0.6468  ...     0.409     0.557     0.398
        # 2    Non-Hispanic Black     0.5131     0.5986  ...     0.330     0.468     0.358
        # 3              Hispanic     0.3688     0.4491  ...     0.260     0.385     0.282
        # 4                 Other     0.3969     0.4625  ...     0.291     0.413     0.310
        # 5                   NaN        NaN        NaN  ...       NaN       NaN       NaN
        # 6   Share of Electorate  2018.0000  2016.0000  ...  1990.000  1988.000  1986.000
        # 7    Non-Hispanic White     0.7326     0.7363  ...     0.856     0.851     0.850
        # 8    Non-Hispanic Black     0.1222     0.1234  ...     0.093     0.097     0.100
        # 9              Hispanic     0.0940     0.0908  ...     0.035     0.036     0.036
        # 10                Other     0.0512     0.0495  ...     0.016     0.017     0.015
        # 11                  NaN        NaN        NaN  ...       NaN       NaN       NaN
        
        # Turn 'Share of Electorate' into columns:
        
        # Remove the rows with nan:
        df = df.drop(df[df.PopSegment.isnull()].index)

        # Drop the intermediate header: "Share of Electorate" 2018.000...
        df = df.drop(df[df.PopSegment == 'Share of Electorate'].index)

        # Don't want the 'Census Weight' versions of these
        # stats. So remove all rows starting at 'Census Weight'
        # in the PopSegment col. The [0] is b/c an
        # array-like is returned:
        
        del_row_start = df[df.PopSegment == 'Census Weight'].index[0]
        df = df.iloc[df.index < del_row_start]
        
        # df's index is now non-standard, starting 
        # with 1. Fix that:
        df = df.reset_index()
        
        # Disambiguate population segment names: 
        new_popsegment_names = pd.Series([
        'TurnoutNonHispWhite',
        'TurnoutNonHispBlack',
        'TurnoutHispanic',
        'TurnoutOther',
        'PercOfElectorateNonHispWhite',
        'PercOfElectorateNonHispBlack',
        'PercOfElectorateHispanic',
        'PercOfElectorateOther'], name='PopSegment')

        # Update works in-place:
        df.update(new_popsegment_names)
        
        # Other election data is only from 2004;
        # discard cols before that:
        
        df = df.drop(list(range(1986,2004,2)), axis=1)

        # Get race/ethicity distributions for each
        # State. Like:
        #                          State  White Black Hispanic,...  Two Or More Races
        # Region Election                 ...                   
        # US     2008      United States  ...               0.02
        # AL     2008            Alabama  ...               0.01
        # AK     2008             Alaska  ...               0.07
        # AZ     2008            Arizona  ...               0.02        
        
        race_population_df = self.import_race_distribution()
        
        # Combine columns other than White,Black,Hispanic into
        # new column: 'Other': sum those cols:

        col_set       = set(race_population_df.columns)
        main_cols_set = set(['State', 'Year', 'White', 'Black', 'Hispanic'])
        other_cols    = col_set.difference(main_cols_set)
        
        # Pacific islanders is tiny, and has too many nan:
        other_cols.remove('Native Hawaiian/Other Pacific Islander')
        
        # Native Americans has some strings: "<.01"
        # Replace those with 0.005
        race_population_df = race_population_df.replace("<.01", 0.005)
        
        # Some numbers in race_population_df are strings.
        # Fix that:
        conversion_dict = {'White': float,
                           'Black': float,
                           'Hispanic': float,
                           }
        for col in other_cols:
            conversion_dict[col] = float
            
        race_population_df = race_population_df.astype(conversion_dict)
        
        # Take the mean column-wise of the thin race cols: 
        other_col_series  = race_population_df[other_cols].mean(axis=1)
        
        # Get the full set of columns to replace
        # again:
        other_cols = col_set.difference(main_cols_set)
        race_population_df = race_population_df.drop(other_cols, axis=1)
        race_population_df['Other'] = other_col_series
        
        # Race populations for some State(s) for some year(s)
        # are NaN. To fix:
        #   o Get each State's rows by themselves, like:
        #                State  Year  White  Black  Hispanic   Other
        #     Group Arkansas:
        #       AK 2004    ...  2004   0.5    0.2    0.5       0.002
        #       AK 2006    ...  2006   0.3    0.5    0.1       0.005
        #       AK 2008    ...  2008   0.5    NaN    0.5       0.006
        #     Group Alabama:
        #       AL 2004    ...  2004   0.5    0.2    0.5       0.002
        #       AL 2006    ...  2006   NaN    0.5    0.1       0.005
        #       AL 2008    ...  2008   0.1    0.6    0.5       0.006
        #
        #   o Within each group, find NaNs and replace them 
        #     with the mean of the non-NaN numbers in the same
        #     column and group. So the NaN in the Black column
        #     will be (0.2 + 0.5) / 2 = 0.35. And the NaN in 
        #     the White column will be (0.5+0.1)/2 = 0.6
        state_groups = race_population_df.groupby(level='Region')
        race_population_df = state_groups.fillna(state_groups.mean())
        
        # We now have:
        #    race_population_df
        #                             State  Year  White Black  Hispanic     Other
        #    Region Election                                                      
        #    US     2008      United States  2008   0.66  0.12      0.16  0.023333
        #    AL     2008            Alabama  2008   0.69  0.26      0.03  0.010000
        #    AK     2008             Alaska  2008   0.66  0.03      0.05  0.083333
        #    AZ     2008            Arizona  2008   0.58  0.03      0.30  0.026667
        #                    ...
        
        # And:
        #     voter_turnout
        #                              State  Year  ...  TotalIneligibleFelons  MeanPastTurnout
        #     Region Election                       ...                                        
        #     US     2004      United States  2004  ...             3158443.00         0.516321
        #     AL     2004            Alabama  2004  ...               52664.00         0.500178
        #     AK     2004             Alaska  2004  ...                8240.00         0.595952
        #     AZ     2004            Arizona  2004  ...               71974.00         0.480923
        #     AR     2004           Arkansas  2004  ...               42885.00         0.469521
        #                    ...

        # And:
        #  self.voting_eligible_population
        #     Region  Election
        #     US      2004        2.034835e+08
        #     AL      2004        3.292608e+06
        #     AK      2004        4.521240e+05
        #     AZ      2004        3.717055e+06
        #                    ...
        
        # For each State, add the fraction of the 
        # voting eligible population for each of 
        # White, Black, Hispanic, and 'Other'

        # We only have race population info back
        # to 2008, so sacrifice 2004 and 2006 at
        # this point:
        
        voter_turnout_trimmed = \
            voter_turnout.drop(voter_turnout[voter_turnout['Year'] < 2008].index, 
                                   axis=0)
        
        for race in ['White','Black','Hispanic','Other']:
            race_pop_fraction = race_population_df[race] * self.voting_eligible_population
            col_name = f'VEP_{race}'
            voter_turnout_trimmed[col_name] = race_pop_fraction

         
        return voter_turnout_trimmed

    #------------------------------------
    # import_race_distribution 
    #-------------------
    
    def import_race_distribution(self):
        '''
        
        Return a df like:
        
                                     State  ...  Two Or More Races
            Region Election                 ...                   
            US     2008      United States  ...               0.02
            AL     2008            Alabama  ...               0.01
            AK     2008             Alaska  ...               0.07
            AZ     2008            Arizona  ...               0.02        


        where the columns are:
            ['State', 'Year', 'White', 'Black', 'Hispanic',
           'American Indian/Alaska Native', 'Asian',
           'Native Hawaiian/Other Pacific Islander', 'Two Or More Races']
       
        and the index is like:
       
            MultiIndex([('US', 2008),
                        ('AL', 2008),
                        ('AK', 2008),
                        ('AZ', 2008),
                          ...
                          
        Data are from the public health site
        https://www.kff.org/other/state-indicator/distribution-by-raceethnicity
        and only go back to 2008.
        
        @return: a df with per-state breakdown of race percentages
           by State. For years 2008/10/12/14/16/18.
        @rtype: pd.DataFrame
        
        '''

        race_df_all_elections = None
        
        for year in self.RACE_ETHNICITY_BY_STATE.keys():
                
            race_df = pd.read_csv(self.RACE_ETHNICITY_BY_STATE[year],
                        skiprows=2,
                        usecols=lambda col_name: col_name != 'notes'
                        )
            # Rows towards the end are notes embedded in
            # the spreadsheet:
            
            race_df = race_df.dropna(axis='index', subset=['Total'])
            
            # Now we have:
            # race_df
            #                 Location  White Black  ...  Two Or More Races Total  Footnotes
            # 0          United States   0.60  0.12  ...               0.03   1.0        1.0
            # 1                Alabama   0.66  0.26  ...               0.02   1.0        NaN
            # 2                 Alaska   0.60  0.03  ...               0.07   1.0        NaN
            # 3                Arizona   0.54  0.04  ...               0.02   1.0        NaN
            # 4               Arkansas   0.72  0.15  ...               0.03   1.0        NaN    
            # 5             California   0.37  0.05  ...               0.03   1.0        NaN        
            #           ...
            
            # Remove the footnotes:
            race_df = race_df.drop('Footnotes', axis=1)
            race_df = race_df.rename({'Location' : 'State'}, axis=1)
            # Add a 'Year' column after the state:
            state_col_int = race_df.columns.get_loc('State')
            race_df.insert(state_col_int+1, 'Year', [year]*len(race_df))
            
            # Puerto Rico cannot vote, remove it:
            race_df = race_df[race_df.State != 'Puerto Rico']
            
            # The 'Total' column is always 1 since all
            # cols are percentages. Remove it:
            
            race_df = race_df.drop('Total', axis=1)
    
            # Create the standard index: State/Election
            # to enable merging with other tables:
            
            state_abbrevs = self.utils.state_abbrev_series(race_df.State)
            region_series         = state_abbrevs
            election_series       = race_df['Year'].copy()
            region_series.name    = 'Region'
            election_series.name  = 'Election'
            race_df = race_df.set_index([region_series, election_series])
            
            if race_df_all_elections is None:
                race_df_all_elections = race_df
            else:
                race_df_all_elections = pd.concat([race_df_all_elections, race_df])

        return race_df_all_elections

    #------------------------------------
    # add_disaster_information 
    #-------------------
    
    def add_disaster_information(self, election_features):
        
        # For convenience of var name brevity:
        df = election_features.copy()
        
        # Sort the index of the df so that we
        # don't get warnings about inefficiency 
        # when setting selected col values:
        df = df.sort_index()
        
        # Initialize the disaster and disaster name cols:
        df['Disaster'] = 0
        df['DisasterName'] = ''

        # We only go back to 2008:
        #df.loc[('LA',2006), 'Disaster'] = 1
        #df.loc[('LA',2006), 'DisasterName'] = 'Katrina'

        # Hurricane Sany, 2012:
        df.loc[('NY',2012), 'Disaster'] = 1
        df.loc[('NY',2012), 'DisasterName'] = 'Sandy'
        df.loc[('NJ',2012), 'Disaster'] = 1
        df.loc[('NJ',2012), 'DisasterName'] = 'Sandy'
        df.loc[('CT',2012), 'Disaster'] = 1
        df.loc[('CT',2012), 'DisasterName'] = 'Sandy'
        df.loc[('VA',2012), 'Disaster'] = 1
        df.loc[('VA',2012), 'DisasterName'] = 'Sandy'
        df.loc[('DE',2012), 'Disaster'] = 1
        df.loc[('DE',2012), 'DisasterName'] = 'Sandy'
        df.loc[('MA',2012), 'Disaster'] = 1
        df.loc[('MA',2012), 'DisasterName'] = 'Sandy'
        df.loc[('NH',2012), 'Disaster'] = 1
        df.loc[('NH',2012), 'DisasterName'] = 'Sandy'

        # Hurricane Matthew, 2016
        df.loc[('FL',2016), 'Disaster'] = 1
        df.loc[('FL',2016), 'DisasterName'] = 'Matthew'
        df.loc[('GA',2016), 'Disaster'] = 1
        df.loc[('GA',2016), 'DisasterName'] = 'Matthew'
        df.loc[('NC',2016), 'Disaster'] = 1
        df.loc[('NC',2016), 'DisasterName'] = 'Matthew'
        df.loc[('SC',2016), 'Disaster'] = 1
        df.loc[('SC',2016), 'DisasterName'] = 'Matthew'

        return df

    #------------------------------------
    # rmse_statewise
    #-------------------
    
    def rmse_statewise(self, predicted, truth):
        '''
        Given two series of voter turnouts by state,
        return RMSE for each state. Expected for
        each of the two series:

             Region Election              
             AK     2008          0.661664
                    2010          0.524586
                    2012          0.524586
                    2014          0.429512
                    2016          0.615751
                    2018          0.534943
             ...                       ...
             WY     2010          0.501196
                    2012          0.501196
                    2014          0.406122
                    2016          0.595712
                    2018          0.514905
                    
        where the column is VoterTurnout (predicted or true)
        
        For each State separately the RMSE between 
        prediction and truth is computed. Returns a 
        Series like:
        
            Region
            AK    0.056036
            AL    0.057293
            AR    0.074859
            AZ    0.050861
            
        @param predicted: predicted voter turnout per State per election
        @type predicted: pd.Series
        @param truth: actual voter turnout per State per election
        @type truth: pd.Series
        @return RMSE between prediction and truth within each State
        '''
        # # Get:
        #                       VoterTurnout  VoterTurnout
        #      Region Election                            
        #      AK     2008          0.661664      0.683000
        #             2010          0.524586      0.529000
        #             2012          0.524586      0.589000
        #             2014          0.429512      0.548000
        #             2016          0.615751      0.614662
        #             2018          0.534943      0.548186
        #      ...                       ...           ...
        #      WY     2010          0.501196      0.460000
        #             2012          0.501196      0.590000
        #             2014          0.406122      0.397000
        #             2016          0.595712      0.602279
        #             2018          0.514905      0.478610

        combo_pred_truth = pd.concat([predicted,truth], axis=1)
        combo_pred_truth.columns = ['TurnoutPred', 'TurnoutTruth']

        # Function to apply when given a dataframe like the 
        # combo above, but with only the data of one State
        # at a time:
        def rmse_by_state(pred_truth_one_state_df):
            rmse = mean_squared_error(pred_truth_one_state_df['TurnoutPred'],
                                      pred_truth_one_state_df['TurnoutTruth'],
                                      squared=False
                                      )
            return rmse

        grp = combo_pred_truth.groupby(by=['Region'])
        
        # Get:
        #     Region
        #     AK    0.056036
        #     AL    0.057293
        #     AR    0.074859
        #          ...

        rmse_df = grp.apply(rmse_by_state)
        return rmse_df

    #------------------------------------
    # target_encode
    #-------------------
    
    def target_encode(self, cat_col, label_col):
        '''
        Given a column with values for a categorical
        variable, and a target col (i.e. the col that is
        to be predicted), return a column with the 
        cat_col values target encoded. 
        
        This scheme has the all occurrences of a
        category value have the same encoding. This
        may lead to data leakage.
         
        @param cat_col: categorical values
        @type cat_col: pd.Series
        @param label_col: values to be predicted
        @type label_col: pd.Series
        @return target encoded values
        @rtype: pd.Series
        '''

        # Sigma adds normal noise to the encodings to 
        # prevent overfitting:
        enc = TargetEncoder(verbose=1, cols=cat_col.name)

        enc = enc.fit(cat_col, label_col)
        # The transform() method calls the encoder's
        # transform_leave_one_out() method:
        res = enc.transform(cat_col, label_col)
        return res

    #------------------------------------
    # binary_encode
    #-------------------
    
    def binary_encode(self, cat_col):
        enc = BinaryEncoder(verbose=1,cols=cat_col.name)
        enc = enc.fit(cat_col)
        res = enc.transform(cat_col)
        return res
    
    #------------------------------------
    # ordinal_encode 
    #-------------------

    def ordinal_encode(self, cat_col):
        '''
        Return a new column where categories
        are encoded as successive integers
        by order of appearance in the given 
        cat_col. Ex.: given ['Mon','Mon','Tue','Wed']
        would return [0,0,1,2]
        
        @param cat_col: column of categical values
        @type cat_col: pd.Series
        @return: series with values replaced
        @rtype: pd.Series
        '''
        
        categories = cat_col.unique()
        replace_dict = {key : val for (key,val) in zip(categories, range(len(categories)))}
        res = cat_col.replace(replace_dict)
        return res

    
    #------------------------------------
    # leave_one_out_encode 
    #-------------------
    
    def leave_one_out_encode(self, cat_col, label_col):
        '''
        Given a column with values for a categorical
        variable, and a target col (i.e. the col that is
        to be predicted), return a column with the 
        cat_col values leave-one-out encoded.
        
        @param cat_col: categorical values
        @type cat_col: pd.Series
        @param label_col: values to be predicted
        @type label_col: pd.Series
        @return leave-one-out encoded values
        @rtype: pd.Series
        '''

        # Sigma adds normal noise to the encodings to 
        # prevent overfitting:
        enc = LeaveOneOutEncoder(verbose=1,
                                 cols=cat_col.name,
                                 sigma=0.05)

        enc = enc.fit(cat_col, label_col)
        # The transform() method calls the encoder's
        # transform_leave_one_out() method:
        res = enc.transform(cat_col, label_col)
        
        return res

    #------------------------------------
    # merge_turnout_query_counts
    #-------------------

    def merge_turnout_query_counts(self, voter_turnout, search_features):
        '''
        Given a voter turnout dataframe, and a df containing
        search query counts, merge the two (i.e. add columns
        of one to the other), and return the result.
        
        @param voter_turnout: dataframe of all voter turnout data
        @type voter_turnout: pd.DataFrame
        @param search_features: dataframe of all voting related search query counts
            by state and election year
        @type search_features: pd.DataFrame
        @return a wider df, containing information from
            both inputs
        @rtype pd.DataFrame
        '''
        # Voter turnout and query frequencies
        # have two column names in common: State, and Year.
        # The rsuffix below renames the search_features
        # columns to State_xxx and Year_xxx in the 
        # join, so we can remove them later:
        
        features = voter_turnout.merge(search_features, 
                                       left_index=True,
                                       right_index=True,
                                       how='inner')

        # Rename/remove columns introduced by
        # the merge where column name ambiguities
        # existed:
        
        features = features.drop(labels=['Year_y'], axis='columns')
        features = features.rename({'Year_x' : 'Year'}, axis='columns')

        return features

# ------------------------ Utilities ----------

    #------------------------------------
    # fold_columns 
    #-------------------
    
    def fold_columns(self, df, cols, new_category_name, new_value_name):
        
        if type(cols) != list:
            cols = [cols]
        
        # The final df will be len(cols) times
        # the current len of df:
        
        non_involved_cols = [col for col in df.columns if col not in cols]
        new_cols = non_involved_cols.copy()
        new_cols.extend([new_category_name, new_value_name])
        
        res = pd.DataFrame([],columns=new_cols)
        
        for col in cols:
            new_col_values   = df[col]
            df_for_col = df[non_involved_cols]
            df_for_col.insert(len(df_for_col.columns),
                              new_category_name,
                              col)
            df_for_col.insert(len(df_for_col.columns),
                              new_value_name,
                              new_col_values)
            res = pd.concat([res, df_for_col])
            
        return res

    #------------------------------------
    # compute_2018_VAPHighestOffice
    #-------------------
    
    def compute_2018_VAPHighestOffice(self):
        '''
        The Election Project's 2018 speadsheet
        is missing the "VAP Highest Office" column.
        This method reads that year's turnout results
        taken from the Census at 
        https://www.census.gov/data/tables/time-series/demo/voting-and-registration/congressional-voting-tables.html
        Table 1.
        
        We read that table, aggregate over all Congressional
        districts to get State level numbers. Then compute the
        missing column, and return it as a pd.Series.
        
        @return one column with 2018 numbers for the 
            VAP Highest Office column.
        @rtype pd.Series
        '''
        turnout_file = os.path.join(self.data_dir, 'votingRatesCongressionalDistricts2018Corrected.xlsx')
        df = pd.read_excel(turnout_file, header=[0], index_col=0)
        df_by_state = df.groupby('StateAbbreviation').sum()
        # Don't need the margins of error:
        voting_rate_VAP = 100 * df_by_state['VotesCast'] / df_by_state['VotingAgePopVAP']

        # Compute the nationwide mean:
        us_mean = voting_rate_VAP.mean()
        voting_rate_VAP = pd.concat([pd.Series(us_mean, index=['US']),voting_rate_VAP])
        
        return voting_rate_VAP


# ------------------------ Main ------------
if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
#                                      formatter_class=argparse.RawTextHelpFormatter,
#                                      description="Predict State from Google queries"
#                                      )
# 
#     parser.add_argument('-l', '--errLogFile',
#                         help='fully qualified log file name to which info and error messages \n' +\
#                              'are directed. Default: stdout.',
#                         dest='errLogFile',
#                         default=None);
# 
#     args = parser.parse_args();

    StatePredictor().run()
    input("Press ENTER to quit...")