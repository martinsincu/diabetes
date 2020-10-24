import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#from sklearn.ensemble import VotingClassifier
import pandas as pd

class Information():

    def __init__(self):
        """
        This class give some brief information about the datasets.
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing values of given datad
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

    def info(self,data):
        """
        print feature name, data type, number of missing values and ten samples of 
        each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        feature_dtypes=data.dtypes
        self.missing_values=self._get_missing_values(data)

        print("=" * 50)

        print("{:16} {:16} {:25} {:16}".format("Feature Name".upper(),
                                            "Data Format".upper(),
                                            "# of Missing Values".upper(),
                                            "Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("="*50)

class Preprocess():

    def __init__(self):
        print("Preprocess object created")

    def fillna(self, data, fill_strategies):
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print("{}: There is no such thing as preprocess strategy".format(strategy))

        return data

    def drop(self, data, drop_strategies):
        for column, strategy in drop_strategies.items():
            data=data.drop(labels=[column], axis=strategy)

        return data

    def feature_engineering(self, data, engineering_strategies=1):
        if engineering_strategies==1:
            return self._feature_engineering1(data)

        return data

    def _feature_engineering1(self,data):

        data=self._base_feature_engineering(data)
        #data = data.loc[data.gender.isin(['Female', 'Male'])]

        drop_strategy = {'examide': 1,  # 1 indicate axis 1(column)
            'citoglipton': 1,
            'metformin-pioglitazone': 1,
            'acetohexamide': 1,
            'metformin-rosiglitazone': 1,
            'glimepiride-pioglitazone': 1,
            'glipizide-metformin': 1,
            'tolbutamide': 1,
            'troglitazone': 1,
            'tolazamide': 1,
            'repaglinide': 1,
            'acarbose': 1,
            'glyburide-metformin': 1,
            'chlorpropamide': 1,
            'nateglinide': 1,
            'miglitol': 1,
            'weight': 1}
        data = self.drop(data, drop_strategy)
        
        vars_top = ['medical_specialty','diag_1', 'diag_2', 'diag_3']
        outliers_top = ['number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient', 'num_medications', 'num_lab_procedures']
        top_n = 6
        
        data = self._top_n_categorics(data, vars_top, top_n)
        data = self._top_outliers(data, outliers_top )
        
        data = self._top_n_categorics(data, ['admission_type_id'], 5)
        data = self._top_n_categorics(data, ['admission_source_id'], 4)
        data = self._top_n_categorics(data, ['discharge_disposition_id'], 6)

        return data

    def _base_feature_engineering(self,data):
        data[['admission_type_id', 'admission_source_id', 'discharge_disposition_id']] = data[['admission_type_id', 'admission_source_id', 'discharge_disposition_id']].astype(str)
        
        data['payer_code'] = data['payer_code'].replace({ '?' : 'rare' })
        data['payer_code'] = np.where( data['payer_code'].isin(['MC','HM','SP','BC','MD','CP','UN','rare']), data['payer_code'], "other")
        data['weight'] = np.where( data['weight']=="?" , "1", "0")
        data['age'] = np.where( data['age'].isin(['[20-30)','[10-20)','[0-10)']), "[0-30)", data['age'])
        data['race'] = np.where( data['race'].isin(['?','Asian','Other','Hispanic']), "Other", data['race'])
        data['metformin'] = np.where( data['metformin']!="Steady", "No", data['metformin'])
        data['max_glu_serum'] = np.where( data['max_glu_serum']==">300", ">200", data['max_glu_serum'])
        #data['A1Cresult'] = np.where( data['A1Cresult']==">8", ">7", data['A1Cresult'])
        data['glimepiride'] = np.where( data['glimepiride']!="Steady", "No", data['glimepiride'])
        data['glipizide'] = np.where( data['glipizide']!="Steady", "No", data['glipizide'])
        data['glyburide'] = np.where( data['glyburide']!="Steady", "No", data['glyburide'])
        data['pioglitazone'] = np.where( data['pioglitazone']!="Steady", "No", data['pioglitazone'])
        data['rosiglitazone'] = np.where( data['rosiglitazone']!="Steady", "No", data['rosiglitazone'])

        return data

    def _label_encoder(self,data):
        labelEncoder=LabelEncoder()
        for column in data.columns.values:
            if 'int64'==data[column].dtype or 'float64'==data[column].dtype or 'int64'==data[column].dtype:
                continue
            labelEncoder.fit(data[column])
            data[column]=labelEncoder.transform(data[column])
        return data
    
    def _top_n_categorics(self, data, vars_top, top_n):
        
        for var in vars_top:
            lista_n= pd.concat( [ ( data.groupby([var])[var].count()/len(data) ).rename("[%]"),
                ( data.groupby([var])[var].count() ).rename("count")], axis=1
                ).sort_values(by='[%]', ascending=False).head(top_n).index.tolist()
            data[var] = np.where( data[var].isin(lista_n), data[var], "other")
        
        return data
    
    def _top_outliers(self, data, vars_top):
        for col in vars_top:
            upper = data[col].quantile(0.98)
            data.loc[data[col]>upper,col] = upper
        
        return data

    def _get_dummies(self, data, prefered_columns=None):

        if prefered_columns is None:
            columns=data.columns.values
            non_dummies=None
        else:
            non_dummies=[col for col in data.columns.values if col not in prefered_columns ]

            columns=prefered_columns


        dummies_data=[pd.get_dummies(data[col],prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)

class PreprocessStrategy():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self):
        self.data=None
        self._preprocessor=Preprocess()

    def strategy(self, data, strategy_type="strategy1"):
        self.data=data
        if strategy_type=='strategy1':
            self._strategy1()
        elif strategy_type=='strategy2':
            self._strategy2()

        return self.data

    def _base_strategy(self):
        #self.data['target'] = np.where( self.data['readmitted']=="<30", 1, 0)
        
        drop_strategy = {'encounter_id': 1,  # 1 indicate axis 1(column)
                         'patient_nbr': 1,
                         'readmitted': 1}
        self.data = self._preprocessor.drop(self.data, drop_strategy)
        
        #self.data = self.data.loc[self.data.gender.isin(['Female', 'Male'])]
        #self.data = self.data.loc[~self.data.discharge_disposition_id.isin([11,13,14,19,20,21])]
        self.data = self.data.replace('?', "missing")

        self.data = self._preprocessor.feature_engineering(self.data, 1)
        #self.data = self._preprocessor._label_encoder(self.data)

    def _strategy1(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=['gender', 'change', 'rosiglitazone', 'pioglitazone', 'glyburide', 'glipizide', 'glimepiride', 'metformin', 'diabetesMed', 'max_glu_serum', 'race', 'insulin', 'A1Cresult', 'admission_source_id', 'admission_type_id', 'medical_specialty', 'discharge_disposition_id', 'diag_1', 'diag_3', 'diag_2', 'age', 'payer_code'])
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        self.data.columns = [regex.sub("|", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in self.data.columns.values]


    def _strategy2(self):
        self._base_strategy()

        #self.data=self._preprocessor._get_dummies(self.data,
        #                                prefered_columns=None)#None mean that all feature will be dummied


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier

class GridSearchHelper():
    def __init__(self):
        print("GridSearchHelper Created")

        self.gridSearchCV=None
        self.clf_and_params=list()

        self._initialize_clf_and_params()

    def _initialize_clf_and_params(self):
        
        clf = CatBoostClassifier()
        params = {'iterations': [500],
            'depth': [4, 6],
            'loss_function': ['Logloss'], # 'CrossEntropy'
            'l2_leaf_reg': np.logspace(-20, -19, 3),
            'leaf_estimation_iterations': [10],
            'logging_level':['Silent'],
            'random_seed': [42]
         }
        self.clf_and_params.append((clf, params))
        
        clf= XGBClassifier()
        params={"learning_rate"    : [0.05, 0.1, 0.2],
                "max_depth"        : [ 3, 5, 7],
                "min_child_weight" : [ 3, 5, 7 ],
                "gamma"            : [ 0.2 , 0.4],
                "colsample_bytree" : [ 0.5 , 0.7]
                }
        self.clf_and_params.append((clf, params))

        clf= KNeighborsClassifier()
        params={'n_neighbors':[5,9],
          'leaf_size':[1,3],
          'weights':['uniform'] # 'distance'
          }
        self.clf_and_params.append((clf, params))

        clf = RandomForestClassifier()
        params = {'n_estimators': [9, 15, 30],
              'max_features': ['sqrt'], # 'log2'
              #'criterion': ['entropy', 'gini'],
              'max_depth': [10, 15],
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1,5]
             }
        #Because of depricating warning for RandomForestClassifier which is not appended.
        #But it give high competion accuracy score. You can append when you run the kernel
        self.clf_and_params.append((clf, params))

    def fit_predict_save(self, X_train, y_train, strategy_type):
        self.X_train=X_train
        self.y_train=y_train
        self.strategy_type=strategy_type
        
        ################# Select Drivers ##########################
        filter_col = self.X_train.columns #selecionar features ###<<<<<<<<<<<<
        #target='target'

        forest = RFC(n_jobs=2,n_estimators=250)
        forest.fit(self.X_train[filter_col], self.y_train)
        importances1 = forest.feature_importances_

        extree = ExtraTreesClassifier()
        extree.fit(self.X_train[filter_col], self.y_train)
        # display the relative importance of each attribute
        relval = extree.feature_importances_

        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(self.X_train[filter_col], self.y_train)
        importances2 = forest.feature_importances_

        top_v = 25
        drivers1 = pd.DataFrame( {'IMP':importances1 ,'Driver':filter_col} 
                    ).sort_values(by=["IMP"],ascending=False).head(top_v)['Driver'].tolist()
        drivers2 = pd.DataFrame( {'IMP':relval ,'Driver':filter_col} 
                    ).sort_values(by=["IMP"],ascending=False).head(top_v)['Driver'].tolist()
        drivers3 = pd.DataFrame( {'IMP':importances2 ,'Driver':filter_col} 
                    ).sort_values(by=["IMP"],ascending=False).head(top_v)['Driver'].tolist()

        drivers = drivers1+drivers2+drivers3
        drivers = set(drivers)
        drivers = list(drivers)
        #print(len(drivers))
        
        self.drivers=drivers
        
        ############
        
        clf_and_params = self.get_clf_and_params()
        models=[]
        self.results={}
        for clf, params in clf_and_params:
            self.current_clf_name = clf.__class__.__name__
            grid_search_clf = GridSearchCV(clf, params, cv=5, scoring='recall') #scoring='recall'
            grid_search_clf.fit(self.X_train[self.drivers], self.y_train)
            clf_train_acc = round(grid_search_clf.score(self.X_train[self.drivers], self.y_train) * 100, 2)
            print(self.current_clf_name, " trained and used for prediction on test data...")
            self.results[self.current_clf_name]=(clf_train_acc, grid_search_clf)
            # for ensemble
            models.append(clf)

            #self.save_result()
            print()
    
    def show_result(self):
        for clf_name, tupla in self.results.items():
            print("{} its recall is {:.3f}".format(clf_name, tupla[0]))

    def get_clf_and_params(self):

        return self.clf_and_params

    def add(self,clf, params):
        self.clf_and_params.append((clf, params))

class ObjectOrientedDiabetes():

    def __init__(self, train):
        """
        :param train: train data will be used for modelling
        :param test:  test data will be used for model evaluation
        """
        print("ObjectOrientedDiabetes object created")
        #properties
        self.number_of_train=train.shape[0]

        self.y_train=train['target']
        self.train=train.drop('target', axis=1)

        #concat train and test data
        self.all_data=self._get_all_data()

        #Create instance of objects
        self._info=Information()
        self.preprocessStrategy = PreprocessStrategy()
        self.gridSearchHelper = GridSearchHelper()

    def _get_all_data(self):
        #return pd.concat([self.train, self.test])
        return self.train

    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self._info.info(self.all_data)

    def preprocessing(self, strategy_type):
        """
        Process data depend upon strategy type
        :param strategy_type: Preprocessing strategy type
        :return:
        """
        self.strategy_type=strategy_type

        self.all_data = self.preprocessStrategy.strategy(self._get_all_data(), strategy_type)


    def machine_learning(self):
        """
        Get self.X_train, self.X_test and self.y_train
        Find best parameters for classifiers registered in gridSearchHelper
        :return:
        """
        self._get_train_and_test()

        self.gridSearchHelper.fit_predict_save(self.X_train,
                                          self.y_train,
                                          self.strategy_type)
    def show_result(self):
        self.gridSearchHelper.show_result()

    def _get_train_and_test(self):
        """
        Split data into train and test datasets
        :return:
        """
        self.X_train=self.all_data[:self.number_of_train]
        #self.X_test=self.all_data[self.number_of_train:]
