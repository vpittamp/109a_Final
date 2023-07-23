import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_regular = pd.read_csv("2021-2022 NBA Player Stats - Regular.csv",encoding='latin-1', delimiter=";")
df_regular.rename(columns={'Rk' : 'Rank', 'Player' : 'Players_name', 'Pos' : 'Position', 'Age' : 'Players_age', 'Tm' : 'Team', 'G' : 'Games_played', 'GS' : 'Games_started', 'MP' : 'Minutes_played_per_game', 'FG' : 'Field_goals_per_game', 'FGA' : 'Field_goal_attempts_per_game', 'FG%' : 'Field_goal_percentage', '3P' : '3_point_field_goals_per_game', '3PA' : '3_point_field_goal_attempts_per_game', '3P%' : '3_point_field_goal_percentage', '2P' : '2_point_field_goals_per_game', '2PA' : '2_point_field_goal_attempts_per_game', '2P%' : '2_point_field_goal_percentage', 'eFG%' : 'Effective_field_goal_percentage', 'FT' : 'Free_throws_per_game', 'FTA' : 'Free_throw_attempts_per_game', 'FT%' : 'Free_throw_percentage', 'ORB' : 'Offensive_rebounds_per_game', 'DRB' : 'Defensive_rebounds_per_game', 'TRB' : 'Total_rebounds_per_game', 'AST' : 'Assists_per_game', 'STL' : 'Steals_per_game', 'BLK' : 'Blocks_per_game', 'TOV' : 'Turnovers_per_game', 'PF' : 'Personal_fouls_per_game', 'PTS' : 'Points_per_game'}, inplace=True)

df_playoffs = pd.read_csv("2021-2022 NBA Player Stats - Playoffs.csv",encoding='latin-1', delimiter=";")

df_playoffs.rename(columns={'Rk' : 'Rank', 'Player' : 'Players_name', 'Pos' : 'Position', 'Age' : 'Players_age', 'Tm' : 'Team', 'G' : 'Games_played', 'GS' : 'Games_started', 'MP' : 'Minutes_played_per_game', 'FG' : 'Field_goals_per_game', 'FGA' : 'Field_goal_attempts_per_game', 'FG%' : 'Field_goal_percentage', '3P' : '3_point_field_goals_per_game', '3PA' : '3_point_field_goal_attempts_per_game', '3P%' : '3_point_field_goal_percentage', '2P' : '2_point_field_goals_per_game', '2PA' : '2_point_field_goal_attempts_per_game', '2P%' : '2_point_field_goal_percentage', 'eFG%' : 'Effective_field_goal_percentage', 'FT' : 'Free_throws_per_game', 'FTA' : 'Free_throw_attempts_per_game', 'FT%' : 'Free_throw_percentage', 'ORB' : 'Offensive_rebounds_per_game', 'DRB' : 'Defensive_rebounds_per_game', 'TRB' : 'Total_rebounds_per_game', 'AST' : 'Assists_per_game', 'STL' : 'Steals_per_game', 'BLK' : 'Blocks_per_game', 'TOV' : 'Turnovers_per_game', 'PF' : 'Personal_fouls_per_game', 'PTS' : 'Points_per_game'}, inplace=True)

mask = df_regular.dtypes == 'object'
categorical = df_regular.columns[mask]
numeric = df_regular.columns[~mask]


# do pca on the numeric data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler.fit(df_regular[numeric])
scaled_data = scaler.transform(df_regular[numeric])

pca = PCA(n_components=5)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# plot the pca components to see how much variance is explained by each component and annotate the variables that contribute the most to each component

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df_regular['Rank'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar()

# plot the cumulative variance explained by the components
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

import seaborn as sns
# plot the heatmap of the correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df_regular[numeric].corr(),cmap='coolwarm',annot=True)


# len(corr_rank)
# len(numeric)

df_regular['Position'].value_counts()

df_regular['Class'] = df_regular['Position'].map({'SG': 'Guard','SF' : 'Forward', 'PG' : 'Guard', 'PF' : 'Forward', 'C' : 'Center','SG-SF' : 'Guard', 'SF-SG' : 'Forward', 'SG-PG' : 'Guard', 'C-PF' : 'Center', 'PF-SF' : 'Forward', 'PG-SG' : 'Guard'})

df_regular.isna().sum().sort_values(ascending=False)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_regular['Class'] = le.fit_transform(df_regular['Class'])

df_regular['Class'].value_counts()

# create new feature for df_regular that is the sum of total points, rebounds, assists, steals, blocks, turnovers, and 3 pointers made

df_regular['Total_stats'] = df_regular['Points_per_game'] + df_regular['Total_rebounds_per_game'] + df_regular['Assists_per_game'] + df_regular['Steals_per_game'] + df_regular['Blocks_per_game'] + df_regular['Turnovers_per_game'] + df_regular['3_point_field_goals_per_game']

df_regular['Total_stats'].describe()
df_regular[df_regular.Total_stats == 0]
# create new features for df_reular that is the points, rebounds, assists, steals, blocks, turnovers, and 3 pointers made per game as a percentage of the 'Total_stats' feature created above for each player in the dataset 

df_regular['Points_per_game_percentage'] = df_regular['Points_per_game'] / df_regular['Total_stats']
df_regular['Total_rebounds_per_game_percentage'] = df_regular['Total_rebounds_per_game'] / df_regular['Total_stats']
df_regular['Assists_per_game_percentage'] = df_regular['Assists_per_game'] / df_regular['Total_stats']
df_regular['Steals_per_game_percentage'] = df_regular['Steals_per_game'] / df_regular['Total_stats']
df_regular['Blocks_per_game_percentage'] = df_regular['Blocks_per_game'] / df_regular['Total_stats']
df_regular['Turnovers_per_game_percentage'] = df_regular['Turnovers_per_game'] / df_regular['Total_stats']
df_regular['3_point_field_goals_per_game_percentage'] = df_regular['3_point_field_goals_per_game'] / df_regular['Total_stats']



# impute missing values with knn imputer

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_regular = imputer.fit_transform(df_regular[numeric])
df_regular = pd.DataFrame(df_regular, columns=numeric)

df_regular.isna().sum().sort_values(ascending=False)

# create train test split of df_regular

from sklearn.model_selection import train_test_split

X = df_regular[numeric].drop(columns=["Class"])
y = df_regular['Class']

# convert y to numeric classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create logistic regression model to classify position

# plot predictors vs. target for each class to see if there is a clear separation between classes 

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].scatter(X_train['Points_per_game'], y_train, c=y_train, cmap='viridis')
ax[0].set_xlabel('Points per game')
ax[0].set_ylabel('Class')

ax[1].scatter(X_train['Total_rebounds_per_game'], y_train, c=y_train, cmap='viridis')
ax[1].set_xlabel('Total rebounds per game')
ax[1].set_ylabel('Class')

plt.show()

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# tune hyperparameters of logistic regression model to improve accuracy score of model on test set (X_test, y_test) using GridSearchCV and cross validation on training set (X_train, y_train) to find best parameters for model

from sklearn.model_selection import GridSearchCV

param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty' : ['l1', 'l2']}
grid = GridSearchCV(logreg, param_grid, cv=5, verbose=3)
grid.fit(X_train, y_train)

grid.best_params_

grid.best_score_

# create new model with best parameters from grid search

logreg2 = LogisticRegression(C=0.001, penalty='l2')
logreg2.fit(X_train, y_train)

y_pred2 = logreg2.predict(X_test)

accuracy_score(y_test, y_pred2)

# create confusion matrix to visualize accuracy of model



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

# create classification report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# use naive bayes to classify position

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy_score(y_test, y_pred)

# create decision tree model to classify position

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

accuracy_score(y_test, y_pred)

# create random forest model to classify position

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

accuracy_score(y_test, y_pred)

# tune hyperparameters of random forest model to improve accuracy score of model on test set (X_test, y_test) using GridSearchCV and cross validation on training set (X_train, y_train) to find best parameters for model

param_grid = {'n_estimators' : [10, 50, 100, 200, 500], 'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid = GridSearchCV(rfc, param_grid, cv=5, verbose=3)
grid.fit(X_train, y_train)

best_max_depth = grid.best_params_['max_depth']
best_n_estimators = grid.best_params_['n_estimators']

grid.best_score_

best_max_depth = grid.best_params_['max_depth']
best_n_estimators = grid.best_params_['n_estimators']
# create new model with best parameters from grid search

rfc2 = RandomForestClassifier(max_depth= best_max_depth, n_estimators=best_n_estimators)
rfc2.fit(X_train, y_train)


y_pred2 = rfc2.predict(X_test)

accuracy_score(y_test, y_pred2)


# create gradient boosting model to classify position

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(n_iter_no_change=10)
hgb.fit(X_train, y_train)

y_pred = hgb.predict(X_test)

accuracy_score(y_test, y_pred)

# tune hyperparameters of gradient boosting model to improve accuracy score of model on test set (X_test, y_test) using GridSearchCV and cross validation on training set (X_train, y_train) to find best parameters for model

param_grid = {'learning_rate' : [0.001, 0.01, 0.1, 1, 10, 100], 'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid = GridSearchCV(hgb, param_grid, cv=5, verbose=3)
grid.fit(X_train, y_train)

grid.best_params_

grid.best_score_

best_max_depth = grid.best_params_['max_depth']
best_n_estimators = grid.best_params_['n_estimators']

# create new model with best parameters from grid search

gbc2 = GradientBoostingClassifier(max_depth= best_max_depth , n_estimators= best_n_estimators)
gbc2.fit(X_train, y_train)
y_pred = gbc2.predict(X_test)

accuracy_score(y_test, y_pred)

# create xgboost model to classify position and tune hyperparameters to improve accuracy score of model on test set (X_test, y_test) using GridSearchCV and cross validation on training set (X_train, y_train) to find best parameters for model

from xgboost import XGBClassifier

# use early stopping to prevent overfitting
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

accuracy_score(y_test, y_pred)

param_grid = {'n_estimators' : [200, 500], 'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'learning_rate' : [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(xgb, param_grid, cv=5, verbose=3)
grid.fit(X_train, y_train)

grid.best_params_

grid.best_score_

best_max_depth = grid.best_params_['max_depth']
best_n_estimators = grid.best_params_['n_estimators']

# create new model with best parameters from grid search

xgb2 = XGBClassifier(max_depth= best_max_depth , n_estimators= best_n_estimators)
xgb2.fit(X_train, y_train)
y_pred = xgb2.predict(X_test)

accuracy_score(y_test, y_pred)
























