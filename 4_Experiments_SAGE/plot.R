library(ggplot2)
library(tidyr) 
library(dplyr)

df = read.csv("./res.csv")

df$OOD_background[grepl("return", df$OOD_background)] = "training_data"
df$OOD_background[grepl("kmeans", df$OOD_background)] = "OOD_kmeans"
df$OOD_background[grepl("knockoff", df$OOD_background)] = "OOD_knockoffs"
df$SAGE_background[grepl("kmeans", df$SAGE_background)] = "SAGE_kmeans"
df$SAGE_background[grepl("return", df$SAGE_background)] = "training_data"

df_n = unique(df$n)
df_p = unique(df$p)

# drop irrelevant columns
df = df %>% select(-num_trees) %>% select(-pert) %>% select(-threshold) %>% 
  select(-break_ooc) %>% select(-n) %>% select(-p) %>% select(-signal_to_noise)



################
# Plot: How often was each variable ranked 1st, 2nd etc.
################
df3 = df %>% 
  gather("variable", "rank", sensitive_feature_rank:x4_rank) %>%
  group_by(cor_strength,OOD_background) %>%
  mutate(mean_fidelity = round(mean(fidelity),4))

df4 = df3 %>% 
  group_by(cor_strength, OOD_background, SAGE_background, SAGE_imputation) %>%
  count(variable, rank) %>% 
  mutate(occurence = n /(sum(n)/df_p))

df4$feature = factor(df4$variable)
df5 = unique(merge(x = df4, y = df3[,c( "cor_strength", "OOD_background", "SAGE_background","SAGE_imputation", "mean_fidelity")], 
                   by.x =  c( "cor_strength", "OOD_background", "SAGE_background", "SAGE_imputation"),
                   by.y = c( "cor_strength", "OOD_background", "SAGE_background", "SAGE_imputation")))
df_plot = df5 %>% filter(OOD_background == "OOD_kmeans")
# full plot for an overview: rankings vs correlation strength
ggplot(data = df_plot, aes(x = rank, y = occurence, fill = feature)) + 
  scale_x_reverse()+
  geom_bar(position = "stack", stat = "identity") + 
  facet_grid(OOD_background + SAGE_background + SAGE_imputation ~ cor_strength+ mean_fidelity) +
 # facet_grid(OOD_background ~ SAGE_background) +
  coord_flip()+
  scale_fill_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  ylab(label = "occurence (%)")+
  theme_minimal()

#-----------------------------------------------------------------------
# This part is code to generate Figure 3
#-----------------------------------------------------------------------

df_plot = df5 %>% 
  filter(cor_strength == 0.5) %>%
  filter(SAGE_background == "SAGE_kmeans") %>%
  filter(OOD_background == "OOD_kmeans")
ggplot(data = df_plot, aes(x = rank, y = occurence, fill = feature)) + 
  scale_x_reverse()+
  geom_bar(position = "stack", stat = "identity") + 
  facet_grid( cor_strength  ~OOD_background + SAGE_background+mean_fidelity + SAGE_imputation) +
  # facet_grid(OOD_background ~ SAGE_background) +
  coord_flip()+
  scale_fill_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  ylab(label = "occurence (%)")+
  theme_minimal()

#-----------------------------------------------------------------------


# Plot other quantities, just out of curiosity
# SAGE value versus cor_strength
df6 = df %>% 
  gather("variable", "value", sensitive_feature:x4) %>%
  group_by(OOD_background, cor_strength) %>%
  mutate(mean_fidelity = round(mean(fidelity),4)) %>%
  ungroup() %>%
  group_by(OOD_background, SAGE_background, SAGE_imputation, variable, cor_strength) %>%
  mutate(mean_value =round(mean(value),4)) %>%
  mutate(sd_value =round(sd(value),4)) 

df_plot = df6 %>% 
  filter(SAGE_background == "SAGE_kmeans") %>%
  filter(OOD_background == "OOD_kmeans")

ggplot(data = df_plot, aes(x = cor_strength, y = mean_value, lty = SAGE_imputation, col = variable)) +
  geom_line()+
  geom_ribbon(aes(ymin= mean_value-sd_value, ymax = mean_value + sd_value, fill = variable), alpha = 0.2, colour = NA)+
  facet_grid(SAGE_imputation~.,scales="free")+
  scale_fill_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  scale_color_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  theme_minimal()
 
# fidelity versus cor_strength

ggplot(data = df_plot, aes(x = cor_strength, y = mean_fidelity)) +
  geom_line()+
  theme_minimal()

## plot mean rank versus correlation strength
df7 = df %>% 
  gather("variable", "rank", sensitive_feature_rank:x4_rank) %>%
  group_by(OOD_background, cor_strength) %>%
  mutate(mean_fidelity = round(mean(fidelity),4)) %>%
  ungroup() %>%
  group_by(OOD_background, SAGE_background, SAGE_imputation, variable, cor_strength) %>%
  mutate(mean_rank =round(mean(rank),4))
df7[df7$SAGE_imputation == "kn", ]$SAGE_imputation = "zkn"
df_plot = df7 %>% 
  filter(SAGE_background == "SAGE_kmeans") %>%
  filter(OOD_background == "OOD_kmeans")


#-----------------------------------------------------------------------
# this code is to generate Figure 4
#-----------------------------------------------------------------------
ggplot(data = df_plot, aes(x = cor_strength, y = mean_rank,  col = variable)) + # lty = SAGE_imputation
  geom_line(size = 2)+
  geom_point(size = 4)+
 # geom_ribbon(aes(ymin= mean_value-sd_value, ymax = mean_value + sd_value, fill = variable), alpha = 0.2, colour = NA)+
  facet_grid(.~SAGE_imputation)+
  scale_fill_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  scale_color_manual(values = c("#F05039", "darkblue", "darkgrey", "lightgrey"))+
  theme_minimal(base_size = 19)+
  scale_y_continuous(breaks = c(1,2,3,4), limits = c(1,4))+
  scale_x_continuous(breaks = unique(df_plot$cor_strength), limits = c(min(df_plot$cor_strength),max(df_plot$cor_strength)))+
 # scale_x_continuous(breaks = c(0,0.1,0.3,0.5,0.7,0.9), limits = c(min(df_plot$cor_strength),max(df_plot$cor_strength)))+
  ylab("mean rank")+
  xlab("correlation")
