#!/usr/bin/env python

"train a binary classifier on extracted features, predict, evaluate"

import pandas as pd

from pprint import pprint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy

#

from config import train_file, test_file

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

x_train = train.drop( 'y', axis = 1 ).values
y_train = train.y.values

x_test = test.drop( 'y', axis = 1 ).values
y_test = test.y.values

classifiers = [
	#LR( C = 10 ),
	#LR( C = 1 ),													
	#LR( C = 0.1 ),									
									
	make_pipeline( StandardScaler(), LR()),	
	#make_pipeline( StandardScaler(), LR( C = 10 )),
	#make_pipeline( StandardScaler(), LR( C = 30 )),

	make_pipeline( MinMaxScaler(), LR()),					
	#make_pipeline( MinMaxScaler(), LR( C = 10 )),	
	#make_pipeline( MinMaxScaler(), LR( C = 30 )),

	#LDA(),										
	RF( n_estimators = 100, min_samples_leaf = 5 )
]

for clf in classifiers:

	clf.fit( x_train, y_train )
	p = clf.predict_proba( x_test )[:,1]
	p_bin = clf.predict( x_test )

	auc = AUC( y_test, p )
	acc = accuracy( y_test, p_bin )
	print( "AUC: {:.2%}, accuracy: {:.2%} \n\n{}\n\n".format( auc, acc, clf ))

