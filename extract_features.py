#!/usr/bin/env python

"extract features, impute nulls, save"

import warnings
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

#

from config import input_file, features_file

if __name__ == '__main__':

	d = pd.read_csv( input_file, header = None )

	columns = list( d.columns )
	columns.pop()
	columns.append( 'target' )
	d.columns = columns

	y = d.target
	d.drop( 'target', axis = 1, inplace = True )

	d = d.stack()
	d.index.rename([ 'id', 'time' ], inplace = True )
	d = d.reset_index()
	
	print len( d )
	print d.head()
	
	# doesn't work too well
	with warnings.catch_warnings():
		warnings.simplefilter( "ignore" )
		f = extract_features( d, column_id = "id", column_sort = "time" )

	#c:\usr\anaconda\lib\site-packages\scipy\signal\spectral.py:1633: 
	# UserWarning: nperseg = 256 is greater than input length  = 152, using nperseg = 152

	# Feature Extraction: 20it [22:33, 67.67s/it]
	
	impute( f )
	assert f.isnull().sum().sum() == 0
	
	f['y'] = y
	f.to_csv( features_file, index = None )
