# functions to impute out-of-coalition values
#' @import data.table
#' @export
knockoff_imputer = function(model, ins, background_distribution, S,  n_kn=parent.frame()$n_kn , batch_ids =parent.frame()$batch_ids, ...){
  S_rep = S[rep(1:nrow(S), each =n_kn),]
  ins_rep = ins[rep(1:nrow(ins), each =n_kn),]
  my_back = background_distribution[id %in% batch_ids,]
  for (c in 1:(ncol(my_back)-1)) data.table::set(my_back,  i=which(S_rep[[c]]==TRUE), j=c, ins_rep[[c]][S_rep[[c]]==TRUE])
  pred_raw = data.table::data.table(cbind("pred" = model(my_back[,setdiff(colnames(my_back),"id"), with=FALSE]), "id" = my_back$id))
  mean_preds = data.table::setDT(pred_raw)[ , .(mean_pred = mean(pred)), by = id]
  return(mean_preds$mean_pred)
}

#' @export
marginal_imputer <-  function(model, background_distribution, S, ins, ...){
  S_rep = S[rep(seq_len(nrow(S)), each =nrow(background_distribution)), ]
  ins_rep = ins[rep(seq_len(nrow(ins)), each =nrow(background_distribution)), ]
  background_distribution_rep = do.call("rbind", replicate(nrow(ins), background_distribution, simplify = FALSE))
  for (c in 1:ncol(background_distribution_rep)) data.table::set(background_distribution_rep,  i=which(S_rep[[c]]==TRUE), j=c, ins_rep[[c]][S_rep[[c]]==TRUE])
  # if regression task - average predictions
  # if classification task - average probs
  #pred_raw = if(is.factor(ins$y[1])){model$predict_newdata(newdata = background_distribution_rep)$prob[,"1"]}else{model$predict_newdata(newdata = background_distribution_rep)$response}
  # pred_raw = if(is.factor(ins$y[1])){predict(model, background_distribution_rep)}else{predict(model, background_distribution_rep)}
  pred_raw = model(background_distribution_rep)
  pred_dt = data.table::data.table("pred" = pred_raw, "id" = rep(1:nrow(ins), each = nrow(background_distribution)))
  mean_preds = data.table::setDT(pred_dt)[ , .(mean_pred = mean(pred)), by = id]
  return(mean_preds$mean_pred)
}
