import os.path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from functions import *

# ------------------------------------------------------------------------------
# Import data
# ------------------------------------------------------------------------------

#dtype for Data/simpsons_script_lines.csv
types = {
    'id' : np.int64,
    'episode_id' : np.int64,
    'number' : np.int64,
    'raw_text' : str,
    'timespamp_in_ms' : np.int64,
    'speaking_line' : str,
    'character_id' : np.float64,
    'location_id'  : np.float64,
    'raw_character_text' : str,
    'raw_location_text' : str,
    'spoken_words' : str,
    'normalized_text': str,
    'word_count': np.float64,
}

dataset_script_lines = pd.read_csv('Data/simpsons_script_lines.csv',
    error_bad_lines=False,
    dtype = types,
)

print 'Dataset info:'
print dataset_script_lines.info()
print

# ------------------------------------------------------------------------------
# Reduce and clean data
# ------------------------------------------------------------------------------
# We keep only the script lines where someone is talking
dataset_script_lines = dataset_script_lines[\
	dataset_script_lines['speaking_line'] == 'true'].dropna()


# Selecting the 4 characters who have the more lines and delete all other entries

selected_characters = \
    dataset_script_lines['raw_character_text'].value_counts()[:4]

to_delete_from_script = []
for index, row in dataset_script_lines.iterrows():
    if row['raw_character_text'] not in selected_characters:
        to_delete_from_script.append(index)

dataset_script_lines = dataset_script_lines.drop(to_delete_from_script)


# Keep only a portion of the original dataset
dataset_script_lines = dataset_script_lines[:10000]

print 'Selected characters:'
print selected_characters
print

del to_delete_from_script
del selected_characters
 #------------------------------------------------------------------------------
# Transform the text into numerical type
# ------------------------------------------------------------------------------

vectorizer = Vectorizer(dataset_script_lines['normalized_text'])

X = vectorizer.transform(dataset_script_lines['normalized_text'])

y = np.asarray(dataset_script_lines.loc[:,'character_id'], dtype=np.float64)

del dataset_script_lines

# ------------------------------------------------------------------------------
# Split data into train set and test set
# ------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.33, random_state=42)

del X
del y

print 'Train size:', X_train.shape
print 'Test size:', X_test.shape
print


# ------------------------------------------------------------------------------
# Training

warnings.simplefilter(action='ignore', category=FutureWarning)

classifiers_id = Load_Classifiers()

# ------------------------------------------------------------------------------
# Training classifiers and printing scores
# ------------------------------------------------------------------------------

for clf_id in classifiers_id:

	clf = TrainCV(clf_id)
	clf.fit(X_train, y_train)
	Save_clf(clf_id, clf, vectorizer)
	Print_Score(clf_id, clf, X_test, y_test)


