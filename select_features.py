#!/usr/bin/env python

"select features, save train and test files"

import pandas as pd
from tsfresh import extract_features, select_features

from config import *

#

# workaround for multiprocessing on windows
if __name__ == '__main__':

	print "loading {}".format( features_file )
	features = pd.read_csv( features_file )

	train_x = features.iloc[:validation_split_i].drop( 'y', axis = 1 )
	test_x = features.iloc[validation_split_i:].drop( 'y', axis = 1 )

	train_y = features.iloc[:validation_split_i].y
	test_y = features.iloc[validation_split_i:].y

	print "selecting features..."
	train_features_selected = select_features( train_x, train_y, fdr_level = fdr_level )

	print "selected {} features.".format( len( train_features_selected.columns ))

	train = train_features_selected.copy()
	train['y'] = train_y

	test = test_x[ train_features_selected.columns ].copy()
	test['y'] = test_y
	
	#

	print "saving {}".format( train_file )
	train.to_csv( train_file, index = None )

	print "saving {}".format( test_file )
	test.to_csv( test_file, index = None )
	