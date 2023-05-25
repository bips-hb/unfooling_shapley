library(dplyr)
library(ggplot2)

# res_adv.csv for adversarial attack of adversarial attack OOD detector omega Slack et al. -- replicating their results
# res_ko.csv for knockoff imputation of adversarial attack OOD detector omega of Slack et al.
# res_ko_ko.csv for knockoff imputation of knockoff trained OOD detector omega
# res_ko_ko_fidelity.csv for knockoff imputation of knockoff trained OOD detector omega, enforcing high fidelity

datasets = c("res_adv.csv", "res_ko.csv", "res_ko_ko.csv", "res_ko_ko_fidelity.csv")
plt_list = list()
for (i in 1:length(datasets)) {
  df= read.csv(datasets[i]) %>% select(-X) 
  df$feature = factor(df$feature, levels = c("Other", "LoanRateAsPercentOfIncome", "Gender"))
  df$feature[is.na(df$feature)] = "Other"
  # numerical values lead to the sum of occurrences to be roughly 1 but not exactly 1  
  # -> adjust values s.t. they sum up to exactly 1
  df =df  %>% group_by(rank)  %>%
    mutate(sums = sum(occurence))  %>% 
    mutate(occurence_adj = occurence/sums) 
  plt_list[[i]] = ggplot(data = df, aes(x = rank, y = occurence_adj, fill = feature)) + 
    scale_x_reverse()+
    geom_bar(position = "stack", stat = "identity") + 
    facet_grid(fidelity + method~ .) +
    coord_flip()+
    scale_fill_manual(values = c("grey", "darkblue", "#F05039"))+
    ylab(label = "occurence (%)")+
    theme_minimal()+
    ggtitle(datasets[i])
}

library(gridExtra)
par(mfrow = c(2,3))
grid.arrange(plt_list[[1]],plt_list[[2]],plt_list[[3]],plt_list[[4]])

#########
# with display of "nothing shown"
# "nothing shown" corresponds to the case where we don't have a ranking between shapley values because they are all zero

datasets = c("res_adv.csv", "res_ko.csv", "res_ko_ko.csv", "res_ko_ko_fidelity.csv")
plt_list = list()
for (i in 1:length(datasets)) {
  df= read.csv(datasets[i]) %>% select(-X) 
  df$feature = factor(df$feature, levels = c("Other", "LoanRateAsPercentOfIncome", "Gender", "Nothing shown"))
  df$feature[is.na(df$feature)] = "Other"
  # numerical values lead to the sum of occurrences to be roughly 1 but not exactly 1  
  # -> adjust values s.t. they sum up to exactly 1
  df =df  %>% group_by(rank)  %>%
    mutate(sums = sum(occurence))  %>% 
    mutate(occurence_adj = occurence/sums)  
  plt_list[[i]] = ggplot(data = df, aes(x = rank, y = occurence, fill = feature)) + 
    scale_x_reverse()+
    geom_bar(position = "stack", stat = "identity") + 
    facet_grid(fidelity + method~ .) +
    coord_flip()+
    scale_fill_manual(values = c("grey", "darkblue", "#F05039", "pink"))+
    ylab(label = "occurence (%)")+
    theme_minimal()+
    ggtitle(datasets[i])
}

par(mfrow = c(2,3))
grid.arrange(plt_list[[1]],plt_list[[2]],plt_list[[3]],plt_list[[4]])




