#' SAGE permutation estimator
#'
#' @param imputer function to impute values for out-of-coalition features; default is marginal imputation
#' @param X input `data.frame` of covariates
#' @param y prediction target
#' @param model trained prediction model that takes an `X` like `data.frame` as input and returns predictions
#' @param loss loss function that takes the actual and predicted value as input and returns loss, e.g. Metrics::logloss()
#' @param n_kn number of knockoffs sampled for each observation in background distribution
#' @param knockoff_fun function to generate knockoffs, should take X as an input and return knockoffs for X
#' @return SAGE values for each feature
#' @import data.table
#' @examples
#' Classification task example:
#' X = iris[, !names(iris)%in%"Species"]
#' y = iris$Species == "setosa"
#' model_fit = glm(y ~ ., data = data.frame(X,y), family = binomial(link = "logit"))
#' model = function(input){predict(model_fit, newdata = input, type = "response")}
#' model_fit = ranger::ranger(y~., data = data.frame(X,y), probability = F)
#' model = function(input){predict(model_fit, data = input, type = "response")$predictions}
#' loss = Metrics::ll
#' SAGE(X = X, y = y, model = model, imputation = "marginal", loss = loss)
#' SAGE(X = X, y = y, model = model, imputation = "knockoff", loss = loss, n_kn = 10, knockoff_fun =  function(x){knockoff::create.second_order(as.matrix(x))})

#' @examples
#' Regression task example
#' model_fit <- lm(Murder ~ ., data=USArrests)
#' model = function(input){predict(model_fit, newdata = input, type = "response")}
#' loss = Metrics::ae
#' X = USArrests[, !names(USArrests)%in%c("Murder")]
#' y = USArrests$Murder
#' knockoff_fun = function(x){knockoff::create.second_order(as.matrix(x))}
#' knockoff_fun = function(x){seqknockoff::knockoffs_seq(data.frame(lapply(x, as.numeric)))}
#' SAGE(X = X, y = y, model = model, imputation = "marginal", loss = loss)
#' SAGE(X = X, y = y, model = model, imputation = "knockoff", loss = loss, n_kn = 10, knockoff_fun =  function(x){seqknockoff::knockoffs_seq(data.frame(lapply(x, as.numeric)))})
#' @export
SAGE <- function(X, y, model, imputation = "marginal", loss, batch_size = 20, n_kn = 1, knockoff_fun = NULL, background_distribution = X, ...){
  if(imputation == "marginal"){
    print("marginal imputation")
    if(nrow(background_distribution) < nrow(X) ){
      background_distribution = background_distribution[sample(1:nrow(X), nrow(X), replace = T),]
    } else {
      background_distribution = background_distribution
    }
    imputer_fun = marginal_imputer
  }
  else if (imputation == "knockoff"){
    print("knockoff imputation")
    # generate knockoffs upfront to ensure good knockoff fit
    background_distribution = do.call(rbind, lapply(replicate(n_kn, X, simplify = F), function(arg){cbind(knockoff_fun(arg), "id"= 1:nrow(arg))}))
    background_distribution = data.table::data.table(background_distribution)
    background_distribution = background_distribution[order(id),]
    imputer_fun = knockoff_imputer
  }
  else (print("Please specify function for imputation"))

  # define batches
  batches = round(seq.int(0, nrow(X), length.out = nrow(X)/batch_size))

  # calculation for each batch
  sage_values <- data.table::data.table(matrix(data = rep(0, length(batches)*ncol(X)), nrow = length(batches), ncol = ncol(X)))
  names(sage_values) = names(X)
  for( i in 2:(length(batches))){
    batch_ids = (batches[i-1]+1):batches[i]
    # select an instance
    ins <- data.table::data.table(X[batch_ids,])
    # initialize SAGE values phi
    phi <- data.table::data.table(matrix(data = rep(0, nrow(ins)*ncol(X)),nrow = nrow(ins), ncol = ncol(X)))
    # sample coalition setup D; subset S of D builds the actual coalition
    perm <- t(apply(matrix(nrow = nrow(ins), ncol = ncol(X)), 1, function(x){x = sample(1:ncol(X), ncol(X))}))
    # calculate initial loss - S = empty set
    S = data.table::data.table(matrix(logical(length(perm)), nrow = nrow(ins)))
    loss_prev <- loss(actual = y[batch_ids], predicted = imputer_fun(model = model,
                                                                     background_distribution = background_distribution,
                                                                     S = S, ins = ins, ...)
                      )

    for(d in 1:ncol(perm)){
      # add feature d to coalition
      for (s in 1:nrow(S)) data.table::set(S, s, perm[,d][s], TRUE)
      # impute values of variables not in S
      y_hat_mean = imputer_fun(model = model, background_distribution = background_distribution, S = S, ins = ins, ...)
      loss_S = loss(actual = y[batch_ids], predicted = y_hat_mean)
      delta = loss_prev - loss_S
      loss_prev = loss_S
      # save importance values phi
      for (p in 1:nrow(phi)) data.table::set(phi, p, perm[,d][p], delta[p])
    }
    means = colMeans(phi)
    for (ss in 1:ncol(sage_values)) data.table::set(sage_values, i = as.integer(i-1), j = ss, means[ss])
    # INSERT HERE: if change to previous phi is small, break -- early stopping
  }
  return(colMeans(sage_values))
}



