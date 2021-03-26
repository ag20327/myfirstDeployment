import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv(r"D:\Descargas\heart.csv")

data.head()

features = ["age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
           "thalach",
           "exang",
           "oldpeak",
           "slope",
           "ca",
           "thal"]
data[features].head()

# Split data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data[features].values,
                                                data.target,
                                                test_size=0.3,
                                                random_state=4)

from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.metrics import accuracy_score as acc
RF = ExtraTreesClassifier(n_estimators=20000, max_depth=4)
RF.fit(Xtrain, Ytrain)
print("ExtraTrees ACC = %.2f" % acc(Ytest, RF.predict(Xtest)))

# Import pickle Package

import pickle

# Save the Modle to file in the current working directory

Pkl_Filename = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Data Science\Lab10\DS_FirstDeployment\myfirstDeployment\Models\Pickle_RF_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(RF, file)


# Import the Logistic Regression Module from Scikit Learn
from sklearn.linear_model import LogisticRegression
# Define the Model
LR_Model = LogisticRegression(C=0.1,
                               max_iter=20,
                               fit_intercept=True,
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)
print("Logistic Regression ACC = %.2f" % acc(Ytest, LR_Model.predict(Xtest)))

# Save the Modle to file in the current working directory

Pkl_Filename = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Data Science\Lab10\DS_FirstDeployment\myfirstDeployment\Models\Pickle_RL_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(LR_Model, file)

# Import the Logistic Regression Module from Scikit Learn
from sklearn.svm import SVC
# Define the Model
SVMC= SVC(C=0.1,
          max_iter=10,
         kernel='rbf',)
# Train the Model
SVMC.fit(Xtrain, Ytrain)
print("SVM Classifier ACC = %.2f" % acc(Ytest, SVMC.predict(Xtest)))

# Save the Modle to file in the current working directory

Pkl_Filename = r"C:\Users\andre\OneDrive\Escritorio\Materias Essex\Data Science\Lab10\DS_FirstDeployment\myfirstDeployment\Models\Pickle_SVMC_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(SVMC, file)