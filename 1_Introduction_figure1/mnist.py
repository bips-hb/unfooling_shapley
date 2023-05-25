import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot
rng = np.random.default_rng(seed=0)

(train_X, train_y), (test_X, test_y) = mnist.load_data()
X = np.concatenate([train_X, test_X])
y = np.concatenate([train_y, test_y])

# recode s.t. mnist is binary
X[X>0] = 1

# some plots to get a first impression of the data set
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.axis('off')
    pyplot.imshow(X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

# prepare deep knockoff sampling 
import pandas as pd
import torch
torch.cuda.empty_cache()
from DeepKnockoffs  import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs

def generate_deep_knockoffs(instance, corr_g):
    torch.set_num_threads(1)

    save_colnames =instance.columns
    instance = np.array(instance)
    training_params = {'LAMBDA':1.0,'DELTA':1.0, 'GAMMA':1.0 }
   # Set the parameters for training deep knockoffs
    pars = dict()
   # Number of epochs
    pars['epochs'] = 10
   # Number of iterations over the full data per epoch
    pars['epoch_length'] = 50
   # Data type, either "continuous" or "binary"
    pars['family'] = "binary"
   # Dimensions of the data
    pars['p'] = instance.shape[1]
  # Size of the test set
    pars['test_size']  = int(0.1*instance.shape[0])
   # Batch size
   # pars['batch_size'] = 64 # works but super slow
    pars['batch_size'] = int(0.45*instance.shape[0])
   # Learning rate
    pars['lr'] = 0.01
   # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
   # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10*instance.shape[1])
   # Penalty for the MMD distance
    pars['GAMMA'] = training_params['GAMMA']
   # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = training_params['LAMBDA']
   # Decorrelation penalty hyperparameter
    pars['DELTA'] = training_params['DELTA']
   # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
   # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]

   # Initialize the machine
    machine = KnockoffMachine(pars)

   # Train the machine
    print("Fitting the knockoff machine...")

    machine.train(instance) 
 
   # Generate deep knockoffs
    Xk_machine = machine.generate(instance)
    Xk_machine = pd.DataFrame(Xk_machine)
    Xk_machine.columns = save_colnames
  
    return(Xk_machine)

def get_target_corr(instance):
    torch.set_num_threads(1)

    save_colnames =instance.columns
    instance = np.array(instance)
    SigmaHat = np.cov(instance, rowvar=False)
    ## Gaussian Knockoffs model
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(instance,int(0)), method="sdp") # use this program to get penalty for pairwise corr
    # knockoffs are not uniquely defined -> get knockoffs that penalize the pairwise corr between variables and their knockoffs -> target corr
    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
    return corr_g

#-----------------------------
# take a digit as an example: digit zero -> corresponds to instance 1
#-----------------------------

my_plot = []

# Original: get original image of digit
ex = X[1]
pyplot.imshow(ex, cmap=pyplot.get_cmap('gray'))
my_plot.append(ex)

# Marginal: sample out-of-coalition rows (second half of the picture) from marginals
ex_second_half = ex.copy()[rng.integers(ex.shape[0], size=ex.shape[0]),14:]
pyplot.imshow(np.concatenate([ex.copy()[:,:14], ex_second_half], axis = 1), cmap=pyplot.get_cmap('gray'))
my_plot.append(np.concatenate([ex.copy()[:,:14], ex_second_half], axis = 1))

# Knockoffs: sample out-of-coalition rows (second halt of the picture) with DeepKnockoffs
ex_ko =   np.concatenate(X[y == 0], dtype=np.int64)
ex_pd = pd.DataFrame(ex_ko).rename(columns=lambda x: "x" + str(x))
target_corr = get_target_corr(ex_pd)
ex_ko =   np.concatenate(X[y == 0], dtype=np.int64)[:(28*30)]      # *number of training samples 
ex_pd = pd.DataFrame(ex_ko).rename(columns=lambda x: "x" + str(x))
kn = generate_deep_knockoffs(ex_pd, corr_g=target_corr)
# important note: in machine.py I dropped the normalization layer after the sigmoid activation function (bug in DeepKnockoffs module code!)
# s.t. I get the probability of belonging to class 1, I here manually set the threshold to 0.5
kn[kn > 0.5] = 1
kn[kn < 0.5] = 0
ex_ko_second_half = kn.to_numpy()[:28, 14:]

# plot example with knockoff half
pyplot.imshow(np.concatenate([ex.copy()[:,:14], ex_ko_second_half], axis = 1), cmap=pyplot.get_cmap('gray'))
my_plot.append(np.concatenate([ex.copy()[:,:14], ex_ko_second_half], axis = 1))

# print final plot
titles = ["Original", "Marginal Sampling", "Knockoff Sampling", "", "", ""]
for i in range(3):  
    pyplot.subplot(330 + 1 + i)
    pyplot.title(titles[i])
    pyplot.axis('off')
    pyplot.imshow(my_plot[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()