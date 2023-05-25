"""
The experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore') 

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
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1, random_state=2022) # random state added by KB
# KB: with validation set
#xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)
#xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.25, random_state=1)

ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)
#xval = ss.transform(xval)

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

##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

#	print ('---------------------')
#	print ("Beginning LIME GERMAN Experiments....")
#	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
#	print ('---------------------')
#
#	# Train the adversarial model for LIME with f and psi 
#	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=categorical)
#	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=categorical)
#                                               
#	explanations = []
#	for i in range(xtest.shape[0]):
#		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())
#
#	# Display Results
#	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
#	print (experiment_summary(explanations, features))
#	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	
	print ('---------------------')
	print ('Beginning SHAP GERMAN Experiments....')
	print ('---------------------')

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

	print ('---------------------')

if __name__ == "__main__":
	experiment_main()

###### KB knockoff version

# now for a different background distribution, namely knockoffs 
pd.DataFrame(xtrain).to_csv("./experiements_SHAP/xtrain.csv")
pd.DataFrame(xtest).to_csv("./experiements_SHAP/xtest.csv")
# run R code for knockoff generation
xtrain_ko = pd.read_csv("./experiements_SHAP/xtrain_ko.csv")
xtest_ko = pd.read_csv("./experiements_SHAP/xtest_ko.csv")
ko_background_distribution = xtrain_ko.to_numpy()
ko_background_distribution_2 = xtest_ko.to_numpy()
ko_adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain,
 feature_names=features, background_distribution=ko_background_distribution,
  rf_estimators=100, n_samples=5e4)
ko_adv_kerenel_explainer = shap.KernelExplainer(ko_adv_shap.predict, ko_background_distribution_2,)
ko_adv_shap.fidelity(xtest)

ko_explanations = ko_adv_kerenel_explainer.shap_values(xtest[1:10])

# format for display
formatted_ko_explanations = []
for exp in ko_explanations:
	formatted_ko_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

print ("SHAP Ranks and Pct Occurances one unrelated features:")
print (experiment_summary(formatted_ko_explanations, features))
print ("Fidelity:",round(ko_adv_shap.fidelity(xtest),2))

# normal adv SHAP
background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
feature_names=features, background_distribution=background_distribution_orig, rf_estimators=100, n_samples=5e4)
adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution_orig,)
explanations = adv_kerenel_explainer.shap_values(xtest[1:10])

	# format for display
formatted_explanations = []
for exp in explanations:
	formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

print ("SHAP Ranks and Pct Occurances one unrelated features:")
print (experiment_summary(formatted_explanations, features))
print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

##### ARCHIVE 

xtrain_ko_2 = pd.read_csv("/Users/kristinblesch/Desktop/slack_fool/experiements_SHAP/xtrain_ko_2.csv")
ko_background_distribution_2 = xtrain_ko_2.to_numpy()
# mix and match kmeans and knockoffs
ko_adv_kerenel_explainer_2 = shap.KernelExplainer(ko_adv_shap.predict, ko_background_distribution_2,)
ko_adv_shap.fidelity(xtest)
ko_explanations = ko_adv_kerenel_explainer_2.shap_values(xtest)
ko_adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), 
innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, 
background_distribution=ko_background_distribution, rf_estimators=10, 
n_samples=500)
dd = ko_adv_shap.ood_training_task_ability
sklearn.metrics.accuracy_score(y_true = dd[0], y_pred = dd[1]) # should be the same as 


## explain one instance
adv_kerenel_explainer.shap_values(xtest[2])
ko_adv_kerenel_explainer.shap_values(xtest[2])
ko_adv_kerenel_explainer_2.shap_values(xtest[2])

ee = ko_adv_shap.perturbation_identifier.predict(xtest_ko.to_numpy())
sum(ee)/ee.shape

# plain RF model
my_rf = RandomForestClassifier().fit(xtrain, ytrain)
my_rf_expl =  shap.KernelExplainer(my_rf.predict, xtrain,)
my_rf_expl.shap_values(xtest[2])

	#Setup SHAP
background_distribution_orig = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, background_distribution=background_distribution_orig, rf_estimators=100, n_samples=5e4)
adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution_orig,)
explanations = adv_kerenel_explainer.shap_values(xtest)
adv_shap.fidelity(xtest)

# get fidelity for this -- using kmeans as background distribution 
adv_shap.fidelity(xtest)
# get OOD ability
cc = ko_adv_shap.ood_training_task_ability
sklearn.metrics.accuracy_score(y_true = cc[0], y_pred = cc[1]) # should be the same as 
ko_adv_shap.fidelity(xtrain_ko.to_numpy())

# merge ko with orig data to see whether conditional setting holds

xnew = np.concatenate([xtrain[:, 0:15], xtrain_ko.to_numpy()[:, 15:29]], axis=1)
ee = adv_shap.perturbation_identifier.predict(ko_background_distribution)
sum(ee)/ee.shape





# results

sum_ko = experiment_summary(formatted_ko_explanations, features)
res_ko = pd.DataFrame()
for i in range(3):
	key = list(sum_ko.keys())
	res_ko = res_ko.append(pd.concat([pd.Series(['knockoff']*len(sum_ko[key[i]])), pd.Series([key[i]]*len(sum_ko[key[i]])), pd.DataFrame(sum_ko[key[i]])], axis=1, ignore_index=False))
res_ko.columns = ['method', 'rank', 'feature', 'occurence']




# plot in python
pp = res_ko.groupby(['feature', 'rank', 'occurence']).size().reset_index().pivot(columns='feature', index='rank', values='occurence')
pp.plot(kind ='barh', stacked=True, ylabel = 'occurence', xlabel = 'rank').invert_yaxis()#.legend(bbox_to_anchor=(1.0, 1.0))