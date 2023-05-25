library(stats)
#------------- 
# helper function to define task
#-------------

models <- function(instance, learner = 'ranger', sensitive_only = T){
  learner = lrn(paste0("regr.", learner), predict_type = "response", num.threads = 1)
  
  # biased model
  if(sensitive_only){
    biased_model = function(data_input){return(data_input$sensitive_feature)}
  }else{
    task = mlr3::as_task_regr(x = instance$train, target = "y")
    biased_model = function(data_input){predict(learner$train(task), data_input, num.threads = 1)}
  }
  # innocent model
  innocent_model = function(data_input){return(data_input$x2)}
  return(list("biased_model" = biased_model, "innocent_model" = innocent_model))
}

#--------------
# OOD classifier: omega
#--------------

#' @param break_ooc: whether to break the out-of-coalition structure, i.e. sample each feature individually. 
#' set TRUE for Slack et al omega
#' set FLASE for knockoff trained omega to maintain correlational structure
#' @param background_distribution: function to generate out-of-coalition features
#' choose function(train_data){kmeans(train_data, 10)$centers for Slack et al omega
#' choose a knockoff function for knockoff imputation, e.g.  function(train_data){knockoff::create.second_order(as.matrix(train_data))}
#' choose background data itself with function(train_data){return(train_data)}
#' set TRUE for knockoff background
#' set FALS for any other background
omega <- function(instance = instance, break_ooc = TRUE, background_distribution = function(train_data){kmeans(train_data, 10)$centers},
                   perturb_multiplier = 2, rf_estimators = 500){
  train = instance$train
  xtrain = train[, !names(train) %in% c("y")]
  xtrain_rep <- xtrain[rep(1:nrow(xtrain),perturb_multiplier ),]
  
  bckgrd_rep = do.call(rbind, replicate(perturb_multiplier, {data.frame(background_distribution(xtrain)) }, simplify = F))
  if(nrow(xtrain_rep) > nrow(bckgrd_rep)){
    print("expand background_distribution to match dimension of input data")
    bckgrd_rep = bckgrd_rep[sample(1:nrow(bckgrd_rep), nrow(xtrain_rep), replace = T), ]
  }
  # slack et al like omega
  if(break_ooc){
    syn = data.frame()
    for (n in 1:nrow(xtrain_rep)) {
      i = sample(1:nrow(xtrain_rep), 1)
      point = xtrain_rep[i,]
      for (p in 1:ncol(xtrain_rep)) {
        j = sample(1:ncol(xtrain_rep), 1)
        point[,j] = bckgrd_rep[sample(1:nrow(bckgrd_rep),1), j]
      }
      syn = rbind(syn, point)
    }
  }
    else{
  # omega that does not break background
  # sample coalitions from power set as mask
  mask <- matrix(data = sample(c(0,1), size = ncol(xtrain_rep)*nrow(xtrain_rep), replace = T), nrow = nrow(xtrain_rep)) == 1
  # synthetic data aka imputed data
  syn = xtrain_rep
  for (c in 1:ncol(syn)){syn[mask[,c], c] = bckgrd_rep[mask[,c],c]}
    }
  
  syn_list <- split(syn, seq(nrow(syn)))
  xtrain_list <- split(xtrain, seq(nrow(xtrain)))
  ys = (syn_list %in% xtrain_list)*1 
  
  all_ys = c(rep(1, nrow(xtrain_rep)), ys)
  all_xs = rbind(xtrain_rep, syn)
  all = cbind("y" = all_ys, all_xs)
  
  # train random forest to distinguish original from imputed data
  is_OOD = ranger::ranger(as.factor(y) ~. , data =all, num.trees = rf_estimators, num.threads = 1, probability = T)
  return(list("omega" = is_OOD))
}


#------------------
# adversarial model
#------------------
adv_pred = function(data_input, OOD_detector, models, thres){
  preds = data.table::fifelse(as.numeric(as.character( predict(OOD_detector$omega, data= data_input, num.threads = 1)$predictions[,1])) >= thres,
                              models$innocent_model(data_input),
                              models$biased_model(data_input))
  return(preds)
}

#------------------
# fooling SAGE
#------------------
library(SAGE)
# NOTE: SAGE is an R package I developed myself. It will soon be available on GitHUB.
# the author/copyright holder is anonymized by "XXX" because of the double blind peer review process
# the version used here can be installed with the files in the SAGE_package_anonoymized folder (is version 0.0.0.9000)


SAGE_wr = function(job, data, instance, OOD_background, SAGE_imputation, num_trees, 
                   pert, break_ooc, threshold, SAGE_background){
  my_models = models(instance = instance, sensitive_only = T ) 
  my_detector = omega(instance = instance, perturb_multiplier = pert, rf_estimators = num_trees, 
                        background_distribution =OOD_background, break_ooc = break_ooc)
  my_adv_model = function(x){adv_pred(x, OOD_detector = my_detector,models = my_models, thres = threshold)}
  if(SAGE_imputation == 'marginal'){
    print('marginal SAGE')
    res = SAGE(model = my_adv_model, y = instance$test$y, X =instance$test[, !names(instance$test) %in% c("y")] , imputation = "marginal", 
               loss = Metrics::se, background_distribution = data.table(SAGE_background(instance$test[, !names(instance$test) %in% c("y")]) ))
  }else if(SAGE_imputation == 'kn'){
    print('knockoff SAGE')
    res =  SAGE(model = my_adv_model, y = instance$test$y, X =instance$test[, !names(instance$test) %in% c("y")] ,
                imputation = "knockoff",n_kn = 10, knockoff_fun = function(x){knockoff::create.second_order(as.matrix(x))},
                loss = Metrics::se) 
  }else{print("ERROR")}
  print(paste0("fidelity on instance$test:", mean( (my_adv_model(instance$test) - my_models$biased_model(instance$test))< .Machine$double.eps ))) 
  res2 = rank(desc(res), ties.method = "max")
  names(res2) = paste0(names(res), "_rank")
  res['fidelity'] =  mean( (my_adv_model(instance$test) - my_models$biased_model(instance$test))< .Machine$double.eps )
  c(res, res2)
}


