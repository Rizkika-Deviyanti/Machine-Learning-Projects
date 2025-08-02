import sys
import matplotlib
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#get data
df = pandas.read_csv("cropdata_updated.csv")

#change data to numeric
d = {'Black Soil' : 0, 'Alluvial Soil' : 1, 'Sandy Soil' : 2, 'Red Soil' : 3, 'Clay Soil' : 4, 'Loam Soil' : 5, 'Chalky Soil' : 6}
df['soil_type'] = df['soil_type'].map(d)

d = {'Wheat' : 0, 'Potato' : 1, 'Carrot' : 2, 'Tomato' : 3, 'Chilli' : 4}
df['crop ID'] = df['crop ID'].map(d)

if df[['soil_type', 'crop ID']].isnull().any().any():
    print("Warning: NaN values detected in 'soil_type' or 'crop ID' after mapping.")
    print(df[['soil_type', 'crop ID']].isnull().sum())
    # Optional: Handle missing values, e.g., drop rows with NaN
    df = df.dropna(subset=['soil_type', 'crop ID'])

#get features for decision tree
features = ['crop ID', 'MOI', 'temp']

dt_majority = df[df['result'] == 0]
dt_minority = df[df['result'] == 1]

#random sampling
sampA = dt_majority.sample(n=1000, random_state=42)
sampB = dt_minority.sample(n=1000, random_state=42)

samp = pandas.concat([sampA, sampB])

#split features and target
X = samp[features]
y = samp['result']

# Normalize features
scaler = MinMaxScaler()
X[features] = scaler.fit_transform(X[features])

#get data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create decision tree
dtree = DecisionTreeClassifier(max_depth=7)
dtree = dtree.fit(X_train, y_train)

#print decision tree
plt.figure(figsize=(70, 50))
tree.plot_tree(dtree,
               feature_names=features,
               filled=False,
               rounded=True,
               label='all',
               class_names=['Not Used', "Used"],
               fontsize=14)
plt.savefig("decision_tree_plot.png")
plt.show()

#predict data from test and count accuracy for decision tree
y_pred = dtree.predict(X_test)
y_pred_train = dtree.predict(X_train)
print("Accuracy : {}".format(accuracy_score(y_test, y_pred)))
print("Train Accuracy : {}".format(accuracy_score(y_train, y_pred_train)))