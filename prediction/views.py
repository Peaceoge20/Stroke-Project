from django.shortcuts import render, HttpResponse
from django.http import HttpResponseBadRequest

## Import required libraries

## warnings
import warnings
warnings.filterwarnings("ignore")

## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn import manifold

## Train-Test Split
from sklearn.model_selection import train_test_split

## Feature selection
from sklearn import feature_selection

## libraraies for classification
from sklearn.pipeline import Pipeline
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score
import io


# Read throuhgh data
df1=pd.read_csv("strokedata12.csv")
df1
df2 = df1.drop(columns=['Study ID', 'LGA','State','PlaceOfBirth','PlaceOfResid', 'FamilyIncome'])
df2
df3 = df2.dropna()
df3


# Number of those who had and don't have stroke
Stokecounts = df3['Stroke'].value_counts()
Stokecounts
X = df3.drop('Stroke', axis = 1)
Y = df3['Stroke']

# train, test
scaler = StandardScaler()
validation_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# test options for classification
num_folds = 10
seed = 7
scoring = 'accuracy'

## spot check the algorithms
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
## Neural Network
models.append(('NN', MLPClassifier()))


results = []
names = []
kfold_results = []
test_results = []
train_results = []
peace1=[]

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(msg)

  # Full Training period
    res = model.fit(X_train, Y_train)
    train_result = accuracy_score(res.predict(X_train), Y_train)
    train_results.append(train_result)

    # Test results
    test_result = accuracy_score(res.predict(X_test), Y_test)
    test_results.append(test_result)
    # print("Name of the Algorithm:", model)
    # print("The Mean of the Cross Validation Score:", cv_results.mean())
    # print("The Standard deviation of the Cross Validation Score:", cv_results.std())
    # print("Accuracy Score for the train model:", train_result)
    # print("Accuracy Score for the test model:", test_result)

    # print("\n")
    # print("CONFUSION MATRIX OF ", model)
    an=[]
    cm = list(confusion_matrix(res.predict(X_test), Y_test))
    for i in cm:
       an.append(list(i))
    cm = an


    # Predict probabilities
    
    validation = np.array2string(cv_results.mean())
    standard =  np.array2string(cv_results.std())
    
    context1 = {
    'model':model,
    'validation': validation,
    'standard': standard,
    'trainresult': train_result,
    'testresult': test_result,
    'conmatrix':cm,
    }
    
    peace1.append(context1)


def pie_chart(lis):  
  # piechart for those who have stroke and don't
  lis=["Have Stroke","No Stroke"]
  Stroke_or_not = df3["Stroke"].value_counts().tolist()
  values = [Stroke_or_not[0], Stroke_or_not[1]]
  fig = px.pie(values=df3["Stroke"].value_counts(), names=lis , width=800, height=400, color_discrete_sequence=["skyblue","black"]
              ,title="percentage between Stroke & No Stroke")
  return HttpResponse(fig.show())

def index(request):
    
        # visualize confusion matrix with seaborn heatmap
      
    context = {
        'peace1':peace1,
    }

    return render(request, 'confusion.html', context)
        
# Plot confusion matrix
def con_matrix(request):
  for name,model in models: 
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type='image/png')

def roccurve(request):
  for name,model in models:
    flag = 0
    try:
      y_score = model.predict_proba(X_test)[:, 1]
    except:
      for x in y_score:
        if x <= 0 :
          flag = 1

    if flag != 1:
      fpr, tpr, thresholds = roc_curve(Y_test, y_score)
      roc_auc = auc(fpr, tpr)

      # Plot the ROC curve
      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend(loc="lower right")
      plt.show()
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close()  # Close the figure to free memory
      buf.seek(0)
      return HttpResponse(buf.getvalue(), content_type='image/png')

    else:
      return HttpResponseBadRequest("ROC curve is not available when  the probability=False for", model) 





# compare algorithms    
def compare(request):
    
    fig = pyplot.figure()
    ind = np.arange(len(names))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
    pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
    fig.set_size_inches(15,8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    pyplot.show()
           
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type='image/png')