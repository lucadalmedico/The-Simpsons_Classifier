import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
import os

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = nltk.WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) \
		for t in word_tokenize(doc)]


def Vectorizer(data):
	filename = 'Utils/Saved_Vectorizer.plk'
		
	if os.path.isfile(filename):
		vectorizer = joblib.load(filename)
	else:
		vectorizer = TfidfVectorizer(
				stop_words='english',
				ngram_range=(1, 2),
				min_df=1,
				#binary=True,
				sublinear_tf=True,
				tokenizer=LemmaTokenizer(),
				)


		vectorizer.fit(data)
		joblib.dump(vectorizer, filename)

	return vectorizer

def Classifier(clf):
	classifiers = {
		'SGDClassifier' : {
			'classifier' : SGDClassifier(),
			'filename' : 'SGDC_saved_model.plk',
			'parameters' : {
		   	 		'loss': ['log','hinge'],
		    			'alpha': [10 ** x for x in range(-6, 1)],
		    			'n_jobs' : [-1],
		   	 		},
			},
		'MultinomialNB' : {
			'classifier' : MultinomialNB(),
			'filename' : 'MNB_saved_model.plk',
			'parameters' : {
					'alpha' : [0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
					'fit_prior' : [True, False],
		   	 		},
			},
		'RandomForestClassifier' : {
			'classifier' : RandomForestClassifier(),
			'filename' : 'RFC_saved_model.plk',
			'parameters' : {
					'n_estimators' : [3, 10, 50],
					'class_weight' : ['balanced',None],
					'n_jobs' : [-1],
		   	 		},
			},
		}

	if clf in classifiers.keys():
		return classifiers[clf]
	else:
		return None

def TrainCV(clf_id):
	try:
		print 'Training ' + clf_id + ' ...' 
		parameters = Classifier(clf_id)['parameters']
		clf = Classifier(clf_id)['classifier']

		clf = GridSearchCV(clf, parameters, \
		 	n_jobs=-1, cv=3, verbose=1, refit=True)

		return clf
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise

def Save_clf(clf_id, clf, vectorizer):
	try:
		pipeline = Pipeline([
		    	('vect', vectorizer),
		    	('clf', clf),
		])

		filename = 'Saved_model/' + Classifier(clf_id)['filename']
		joblib.dump(pipeline, filename)
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise


def Print_Score(clf_name, clf, X_test, y_test):
	try:
		print
		print clf_name,':'
		print
		print clf.best_estimator_
		print
		print 'Best parameters:', clf.best_params_
		print "Validation score:", clf.best_score_	
		print 'Accuracy:', clf.score(X_test, y_test)
		print

		y_pred = clf.predict(X_test)
		
		confusion_matrix = ConfusionMatrix(y_test, y_pred)
		print("Confusion matrix:\n%s" % confusion_matrix)

		print
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise

def Load_Classifiers():
	try:
		classifiers_file = open('Utils/Classifiers.txt', 'r')

		classifiers_id = []
		if classifiers_file:
			classifiers_id = classifiers_file.read().split('\n')
			classifiers_id.remove('')
		return classifiers_id
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise

