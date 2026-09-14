import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load features and subjects 
features = np.load('p_indep_baseline_gaze_resnet50_face224_ssl_sma_baseline_gaze_0.0.npy')
subjects = np.load('subjects_baseline_gaze_resnet50_face224_ssl_sma_baseline_gaze_0.0.npy')

# split data to train and test for training a random forest classifier
X_train, X_test, y_train, y_test = train_test_split(features, subjects, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
# train the classifier
clf.fit(X_train, y_train)
# get the prediction of the subjects on the training and test data
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print(accuracy_score(y_pred_train, y_train))
print(accuracy_score(y_pred_test, y_test))
print(confusion_matrix(y_train, y_pred_train))
print(confusion_matrix(y_pred_test, y_test))
# plot confusion matrix for test data
plot_confusion_matrix(clf, X_test, y_test, normalize='true', values_format='.2f') 
plt.savefig("conf_matrix_indep.png")