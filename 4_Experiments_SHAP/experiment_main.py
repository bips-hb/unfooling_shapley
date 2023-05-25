# This code replicates and extends the German Credit experiment from 
# Slack et al. 2020 "Fooling LIME and SHAP : Adversarial Attacks on Post hoc Explanation Methods"
# with knockoff imputation and we quote their code for the experimental setup from 
# https://github.com/dylan-slack/Fooling-LIME-SHAP 
 

import os
os.chdir('./slack_2020_code')

#--------------------
# START quotation of code from Slack et al. 2020
#--------------------

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans 

from copy import deepcopy

# Set up experiment parameters
params = Params("./model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1, random_state=2022) # random state not in Slack et al. code -- added by KB for reproducibility


ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])


categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone','CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200','SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500','MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere','OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor','OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1','YearsAtCurrentJob_geq_4','JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

#Setup SHAP
background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
		feature_names=features, background_distribution=background_distribution, rf_estimators=100, n_samples=5e4)
adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution,)
explanations = adv_kerenel_explainer.shap_values(xtest)

# format for display
formatted_explanations = []
for exp in explanations:
	formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

print ("SHAP Ranks and Pct Occurances one unrelated features:")
print (experiment_summary(formatted_explanations, features))
print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

#-------------------
# END quotation of code from Slack et al. 2020: 
#-------------------

# define a function to store results 
def store_res(explanations, features, method_name, adv_model):
    formatted_explanations = []
    for exp in explanations:
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    sum = experiment_summary(formatted_explanations, features)
    res = pd.DataFrame()
    key = list(sum.keys())
    for i in range(3):
        res = res.append(pd.concat([pd.Series([method_name]*len(sum[key[i]])), pd.Series([key[i]]*len(sum[key[i]])), pd.DataFrame(sum[key[i]]), pd.Series([adv_model.fidelity(xtest)]*len(sum[key[i]]))], axis=1, ignore_index=False))
    res.columns = ['method', 'rank', 'feature', 'occurence', 'fidelity']
    return(res)

# store results from slack paper
res_adv = store_res(explanations=explanations, features=features, method_name='slack_adv', adv_model=adv_shap)
# write to csv for plotting in R later
os.chdir("..")
pd.DataFrame(res_adv).to_csv("res_adv.csv")

#------------------
# evaluate adversarial model with knockoff imputation
# we here need sequential knockoffs because the data consists of both continuous and categorical variables 
# sequential knockoffs are implemented in R only, hence we here write/read .csv files
#------------------
 
pd.DataFrame(xtrain).to_csv("xtrain.csv")
pd.DataFrame(xtest).to_csv("xtest.csv")
input("Run R code generate_seq_knockoffs.R and then press enter ")

xtrain_ko = pd.read_csv("xtrain_ko.csv").to_numpy()
#xtest_ko = pd.read_csv("./experiments_SHAP/xtest_ko.csv")
xtest_10_ko = pd.read_csv("xtest_10_ko.csv").to_numpy().reshape(10,99,28)

# knockoffs cannot be calculated for single-valued columns 
# -> no knockoffs available for observation number 28, xtest$X8 = 4.54556 --> drop from xtest
xtest = np.delete(xtest, (28), axis = 0)

#-------------------------------------------------------------------
# use the same adversarial classifier, but impute SHAP with knockoffs

res = np.empty((0,xtest.shape[1]), float)
for i in range(xtest.shape[0]):
    xtest_s = xtest[i:(i+1),:]
    ko_background_distribution =xtest_10_ko[:,i,:]
    ko_explainer_s= shap.KernelExplainer(adv_shap.predict, ko_background_distribution)
    ko_explanations_s = ko_explainer_s.shap_values(xtest_s)
    res = np.vstack([res, ko_explanations_s])

res_ko = store_res(explanations=res, features=features, method_name='slack_adv_ko', adv_model=adv_shap)

# write to csv for plotting in R later
pd.DataFrame(res_ko).to_csv("res_ko.csv")

#----------------------------------------------------------------------
# use an adversarial classifier that is trained on knockoff background data
adv_ko_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
		feature_names=features, background_distribution=xtrain_ko, rf_estimators=100, n_samples=5e4)

res = np.empty((0,xtest.shape[1]), float)
for i in range(xtest.shape[0]):
    xtest_s = xtest[i:(i+1),:]
    ko_background_distribution =xtest_10_ko[:,i,:]
    ko_explainer_s= shap.KernelExplainer(adv_ko_shap.predict, ko_background_distribution)
    ko_explanations_s = ko_explainer_s.shap_values(xtest_s)
    res = np.vstack([res, ko_explanations_s])

res_ko_ko = store_res(explanations=res, features=features, method_name='ko_adv_ko', adv_model=adv_ko_shap)
pd.DataFrame(res_ko_ko).to_csv("res_ko_ko.csv")

#----------------------------------------------------------------------
# use an adversarial classifier that is trained on knockoff background data, but this time with other hyperparameters of the random forest
adv_ko_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
		feature_names=features, background_distribution=xtrain_ko, rf_estimators=10, n_samples=500)

res = np.empty((0,xtest.shape[1]), float)
for i in range(xtest.shape[0]):
    xtest_s = xtest[i:(i+1),:]
    ko_background_distribution =xtest_10_ko[:,i,:]
    ko_explainer_s= shap.KernelExplainer(adv_ko_shap.predict, ko_background_distribution)
    ko_explanations_s = ko_explainer_s.shap_values(xtest_s)
    res = np.vstack([res, ko_explanations_s])

res_ko_ko = store_res(explanations=res, features=features, method_name='ko_adv_ko', adv_model=adv_ko_shap)
pd.DataFrame(res_ko_ko).to_csv("res_ko_ko_fidelity.csv")

