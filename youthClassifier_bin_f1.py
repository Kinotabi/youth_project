import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import plot_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import miceforest as mf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate

np.random.seed(42)

frame = os.path.join('/Users/mindonghwan/code/youth/youth_dataset_bin.csv')
df = pd.read_csv(frame)
print(df)

maxIter = 10000

# Create kernels. 
kernel = mf.MultipleImputedKernel(
  data=df,
  save_all_iterations=True,
  random_state=1991
)

# Run the MICE algorithm for 10 iterations on each of the datasets
kernel.mice(10,verbose=True)

new_data = df
# Make a multiple imputed dataset with our new data
new_data_imputed = kernel.impute_new_data(new_data)
# Return a completed dataset
new_completed_data = new_data_imputed.complete_data(0)
X = new_completed_data.drop(['final_class'], axis = 1)
y = new_completed_data['final_class']
classle = LabelEncoder()
y = classle.fit_transform(df['final_class'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, stratify = y)

cross_val_num = 10
cross_val_num_score = 10
iter_num = 100
skf = StratifiedKFold(n_splits = cross_val_num_score, shuffle=True, random_state = 0)


print("=======Logistic regression with l2=======")
#logistic regression
log_cl = LogisticRegression(max_iter = maxIter, penalty='l2')
log_cl.fit(X_train, y_train)
scores_log = cross_val_score(log_cl, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_log}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_log)}")

print("=======Support vector machine=======")
#support vector machine
svm_cl = SVC()
dists_svm = {
 'C': [0.1, 1, 10, 100, 1000], 
 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
 'kernel': ['rbf']
}
clf_svm = RandomizedSearchCV(
    svm_cl,
    dists_svm, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_svm.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_svm.best_params_)

svm_best = clf_svm.best_estimator_
scores_svm = cross_val_score(svm_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_svm}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_svm)}")


print("=======Adaboost=======")
#Adaboost
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
ada_cl = AdaBoostClassifier(base_estimator = DTC)
dists_ada = {
  "base_estimator__criterion" : ["gini", "entropy"],
  "base_estimator__splitter" :   ["best", "random"],
  "n_estimators": [1, 2]
}
clf_ada = RandomizedSearchCV(
    ada_cl,
    dists_ada, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_ada.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_ada.best_params_)

ada_best = clf_ada.best_estimator_
scores_ada = cross_val_score(ada_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_ada}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_ada)}")

print("=======Gradient boosting=======")
#Gradientboost
gb_cl = GradientBoostingClassifier()
dists_gb = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[1, 5, 10, 100]
}
clf_gb = RandomizedSearchCV(
    gb_cl,
    dists_gb, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_gb.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_gb.best_params_)

gb_best = clf_gb.best_estimator_
scores_gb = cross_val_score(gb_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_gb}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_gb)}")


print("=======Random forest=======")
#random forest
rf_cl = RandomForestClassifier()
dists_rf = {
  'bootstrap': [True, False],
  'n_estimators': [200, 500],
  'max_features': ['auto', 'sqrt', 'log2'],
  'max_depth' : [4,5,6,7,8,10,50,100],
  'criterion' :['gini', 'entropy']
}
clf_rf = RandomizedSearchCV(
    rf_cl,
    dists_rf, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_rf.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_rf.best_params_)

rf_best = clf_rf.best_estimator_
scores_rf = cross_val_score(rf_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_rf}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_rf)}")

print(f"Random forest: 한 번씩 검증 때마다 f1 : {scores_rf}")
print(f"Random forest: 50번 모두 검증한 f1 총 평균 : {np.mean(scores_rf)}")
print()

print("=======LightGBM=======")
#LightGBM
lgbm_cl = LGBMClassifier()
dists_lgbm = {
  'learning_rate': [0.01], 
  'n_estimators': [8, 24],
  'num_leaves': [6, 8, 12, 16], 
  'boosting_type': ['gbdt'], 
  'objective': ['binary'], 
  'seed': [500],
  'colsample_bytree': [0.65, 0.75, 0.8], 
  'subsample': [0.7, 0.75], 
  'reg_alpha': [1, 2, 6],
  'reg_lambda': [1, 2, 6],
}
clf_lgbm = RandomizedSearchCV(
    lgbm_cl,
    dists_lgbm, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_lgbm.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_lgbm.best_params_)

lgbm_best = clf_lgbm.best_estimator_
scores_lgbm = cross_val_score(lgbm_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_lgbm}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_lgbm)}")

print("=======XGBboost=======")
#XGBboost
xgb_cl = XGBClassifier(use_label_encoder =False, objective = "binary:logistic")
dists_xgb = {
  'learning_rate': [0.01,0.05,0.1],
  'min_child_weight': [1, 5, 10],
  'gamma': [0.5, 1, 1.5, 2, 5],
  'subsample': [0.6, 0.8, 1.0],
  'colsample_bytree': [0.6, 0.8, 1.0],
  'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
  "n_estimators": [100, 200, 300, 400, 500],
}
clf_xgb = RandomizedSearchCV(
    xgb_cl,
    param_distributions = dists_xgb, scoring='f1', n_jobs = 12,  
    cv=cross_val_num, return_train_score=True, verbose = 3, n_iter=iter_num
)
clf_xgb.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', clf_xgb.best_params_)

xgb_best = XGBClassifier(use_label_encoder =False,
  subsample = 0.6,   
  objective= "binary:logistic", 
  n_estimators= 200, 
  min_child_weight= 1, 
  max_depth= 7, 
  learning_rate= 0.05, 
  gamma= 1, 
  colsample_bytree= 0.6)
scores_xgb = cross_val_score(xgb_best, X_train, y_train, scoring='f1', cv=skf, verbose = 10)
print(f"한 번씩 검증 때마다 f1 : {scores_xgb}")
print()
print(f"5번 모두 검증한 f1 총 평균 : {np.mean(scores_xgb)}")

#overall results
print("=======OVERALL=======")

print(f"logistic with l2: 한 번씩 검증 때마다 f1 : {scores_log}")
print(f"logistic with l2: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_log)}")
print()


print(f"SVM: 한 번씩 검증 때마다 f1 : {scores_svm}")
print(f"SVM: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_svm)}")
print()

print(f"Random forest: 한 번씩 검증 때마다 f1 : {scores_rf}")
print(f"Random forest: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_rf)}")
print()

print(f"Adaboost: 한 번씩 검증 때마다 f1 : {scores_ada}")
print(f"Adaboost: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_ada)}")
print()

print(f"Gradient boosting: 한 번씩 검증 때마다 f1 : {scores_gb}")
print(f"Gradient boosting: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_gb)}")
print()

print(f"LightGBM: 한 번씩 검증 때마다 f1 : {scores_lgbm}")
print(f"LightGBM: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_lgbm)}")
print()

print(f"XGBoost: 한 번씩 검증 때마다 f1 : {scores_xgb}")
print()
print(f"XGBoost: 10번 모두 검증한 f1 총 평균 : {np.mean(scores_xgb)}")



#SHAP
X100 = shap.utils.sample(X, 100)
explainer = shap.Explainer(log_cl.predict, X100)
shap_values = explainer(X_test)

print(f'Current label Shown: 1')
shap.plots.beeswarm(shap_values, max_display=14)



importance = log_cl.coef_[0]
#importance is a list so you can plot it. 
feat_importances = pd.Series(importance)
feat_importances.nlargest(20).plot(kind='barh',title = 'Feature Importance')
plt.show()

