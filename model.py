{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3b9c430",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7dc2796a",
   "metadata": {},
   "outputs": [],
   "source": [
    "data0 = pd.read_csv('urldata.csv')\n",
    "data0.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2c673e14",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'data0' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mdata0\u001b[49m\u001b[38;5;241m.\u001b[39mshape\n",
      "\u001b[1;31mNameError\u001b[0m: name 'data0' is not defined"
     ]
    }
   ],
   "source": [
    "data0.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8286567",
   "metadata": {},
   "outputs": [],
   "source": [
    "data0.columns\n",
    "data0.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d01025f",
   "metadata": {},
   "outputs": [],
   "source": [
    "data0.hist(bins = 50,figsize = (15,15))\n",
    "plt.show()\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae948ebd",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(15,13))\n",
    "sns.heatmap(data0.corr())\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55f69691",
   "metadata": {},
   "outputs": [],
   "source": [
    "data0.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2907cb4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data0.drop(['Domain'], axis = 1).copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "261ee134",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "daf2892b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed\n",
    "data = data.sample(frac=1).reset_index(drop=True)\n",
    "data.head()\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae404efe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sepratating & assigning features and target columns to X & y\n",
    "y = data['Label']\n",
    "X = data.drop('Label',axis=1)\n",
    "X.shape, y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ef53370",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the dataset into train and test sets: 80-20 split\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, \n",
    "                                                    test_size = 0.2, random_state = 12)\n",
    "X_train.shape, X_test.shap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7702c311",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "     \n",
    "\n",
    "# Creating holders to store the model performance results\n",
    "ML_Model = []\n",
    "acc_train = []\n",
    "acc_test = []\n",
    "\n",
    "#function to call for storing the results\n",
    "def storeResults(model, a,b):\n",
    "  ML_Model.append(model)\n",
    "  acc_train.append(round(a, 3))\n",
    "  acc_test.append(round(b, 3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b8b96b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Decision Tree model \n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# instantiate the model \n",
    "tree = DecisionTreeClassifier(max_depth = 5)\n",
    "# fit the model \n",
    "tree.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d379b774",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ce0fda9",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test_tree = tree.predict(X_test)\n",
    "y_train_tree = tree.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f02cf3c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#computing the accuracy of the model performance\n",
    "acc_train_tree = accuracy_score(y_train,y_train_tree)\n",
    "acc_test_tree = accuracy_score(y_test,y_test_tree)\n",
    "\n",
    "print(\"Decision Tree: Accuracy on training Data: {:.3f}\".format(acc_train_tree))\n",
    "print(\"Decision Tree: Accuracy on test Data: {:.3f}\".format(acc_test_tree))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25a1fa32",
   "metadata": {},
   "outputs": [],
   "source": [
    "#checking the feature improtance in the model\n",
    "plt.figure(figsize=(9,7))\n",
    "n_features = X_train.shape[1]\n",
    "plt.barh(range(n_features), tree.feature_importances_, align='center')\n",
    "plt.yticks(np.arange(n_features), X_train.columns)\n",
    "plt.xlabel(\"Feature importance\")\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f65380f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#storing the results. The below mentioned order of parameter passing is important.\n",
    "#Caution: Execute only once to avoid duplications.\n",
    "storeResults('Decision Tree', acc_train_tree, acc_test_tree)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff0c452b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random Forest model\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# instantiate the model\n",
    "forest = RandomForestClassifier(max_depth=5)\n",
    "\n",
    "# fit the model \n",
    "forest.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91d539a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#predicting the target value from the model for the samples\n",
    "y_test_forest = forest.predict(X_test)\n",
    "y_train_forest = forest.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "041dd895",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#computing the accuracy of the model performance\n",
    "acc_train_forest = accuracy_score(y_train,y_train_forest)\n",
    "acc_test_forest = accuracy_score(y_test,y_test_forest)\n",
    "\n",
    "print(\"Random forest: Accuracy on training Data: {:.3f}\".format(acc_train_forest))\n",
    "print(\"Random forest: Accuracy on test Data: {:.3f}\".format(acc_test_forest))\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aabae5cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#checking the feature improtance in the model\n",
    "plt.figure(figsize=(9,7))\n",
    "n_features = X_train.shape[1]\n",
    "plt.barh(range(n_features), forest.feature_importances_, align='center')\n",
    "plt.yticks(np.arange(n_features), X_train.columns)\n",
    "plt.xlabel(\"Feature importance\")\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b2575fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#storing the results. The below mentioned order of parameter passing is important.\n",
    "#Caution: Execute only once to avoid duplications.\n",
    "storeResults('Random Forest', acc_train_forest, acc_test_forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58c7a987",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Support vector machine model\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "# instantiate the model\n",
    "svm = SVC(kernel='linear', C=1.0, random_state=12)\n",
    "#fit the model\n",
    "svm.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b171789",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predicting the target value from the model for the samples\n",
    "y_test_svm = svm.predict(X_test)\n",
    "y_train_svm = svm.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d85463ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "#computing the accuracy of the model performance\n",
    "acc_train_svm = accuracy_score(y_train,y_train_svm)\n",
    "acc_test_svm = accuracy_score(y_test,y_test_svm)\n",
    "\n",
    "print(\"SVM: Accuracy on training Data: {:.3f}\".format(acc_train_svm))\n",
    "print(\"SVM : Accuracy on test Data: {:.3f}\".format(acc_test_svm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fca066f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#creating dataframe\n",
    "results = pd.DataFrame({ 'ML Model': ML_Model,    \n",
    "    'Train Accuracy': acc_train,\n",
    "    'Test Accuracy': acc_test})\n",
    "results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d35b3224",
   "metadata": {},
   "outputs": [],
   "source": [
    "results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
