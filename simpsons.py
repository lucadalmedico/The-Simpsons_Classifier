import os.path
import sys

from sklearn.externals import joblib
from functions import Classifier, Load_Classifiers

if len(sys.argv) < 2:
    print "Usage: python simpson.py \"sentence\""
    sys.exit(1)

sentence = [sys.argv[1]]

character_id = { 
	1 : 'Marge Simpson',
	2 : 'Homer Simpson',
	8 : 'Bart Simpson',
	9 : 'Lisa Simpson',
	}

classifiers_id = Load_Classifiers()

for clf_id in classifiers_id:
	print clf_id

	filename = 'Saved_model/' + Classifier(clf_id)['filename']
	
	clf = joblib.load(filename)

	predicted_id = int(clf.predict(sentence))
	print '->', character_id[predicted_id]
	print
