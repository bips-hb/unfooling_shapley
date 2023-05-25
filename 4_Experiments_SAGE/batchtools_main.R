library(batchtools)
library(dplyr)
library(mlr3learners)
library(SAGE)
Sys.setenv(OMP_NUM_THREADS = "1")
RhpcBLASctl::omp_set_num_threads(1)
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
RhpcBLASctl::blas_get_num_procs()

Sys.setenv(MKL_NUM_THREADS = "1")
options(Ncpus = 1)
#options(Ncpus = 24)
library(data.table)

getDTthreads(verbose = getOption("datatable.verbose"))
setDTthreads(threads = 1, restore_after_fork = FALSE,  throttle = NULL)
getDTthreads(verbose = getOption("datatable.verbose"))
set.seed(2022)

# Registry ----------------------------------------------------------------
reg_name <- "sim_batchtools"
reg_dir <- file.path("registries", reg_name)
unlink(reg_dir, recursive = TRUE) # comment this line out when running final version on cluster
makeExperimentRegistry(file.dir = reg_dir, conf.file = "config.R")


# Problems -----------------------------------------------------------
source("problems.R") # DGP

my_dgp <- function(job, data, n, p, ...) {
  generated_dat <-  dgp(n = n, p = p, ...)
  train_instances <- sample(1:nrow(generated_dat), n*0.66, replace = F)
  list(train = generated_dat[train_instances,], 
       test = generated_dat[-train_instances,])
}

addProblem(name = "dgp", fun = my_dgp, seed = 1)

# Algorithms -----------------------------------------------------------
source("algorithms.R") # Algorithms
addAlgorithm(name = "sage_wrapper", fun = SAGE_wr)

# Parameters --------------------------------------------------------------


repls <-  500

cor_strength <- seq(0,0.9,by=0.1)
n <- c(1000)
p = 4
signal_to_noise <- c(2)
pert = c(2)
num_trees = c(500)
#bckgrd_traindat = function(train_data){return(train_data)} # note that suf false and training data background will always predict in-distribution
bckgrd_kmeans = function(train_data){kmeans(train_data, 10)$centers}
bckgrd_kn = function(train_data){knockoff::create.second_order(as.matrix(train_data))}
bckgrd_data = function(train_data){return(train_data)}
#OOD_background = c(bckgrd_kmeans,bckgrd_kn)
OOD_background = c(bckgrd_kmeans, bckgrd_data)
SAGE_imputation= c("marginal", "kn")
SAGE_background = c(bckgrd_kmeans, bckgrd_data)
break_ooc = c(TRUE)
threshold = c(0.5)
# Experiments -----------------------------------------------------------
prob_design <- list(
  dgp = rbind(expand.grid(n = n,
                          p = p,
                          signal_to_noise = signal_to_noise,
                          cor_strength = cor_strength))
)
algo_design <- list(
  sage_wrapper = expand.grid(OOD_background=OOD_background,
                             SAGE_imputation = SAGE_imputation, 
                             pert =pert, 
                             num_trees =num_trees,
                             break_ooc = break_ooc, 
                             threshold = threshold, 
                             SAGE_background = SAGE_background)
)

addExperiments(prob_design, algo_design, repls = repls)
summarizeExperiments()

# Test jobs -----------------------------------------------------------
#testJob(id = 5)

# Submit -----------------------------------------------------------
if (grepl("node\\d{2}|bipscluster", system("hostname", intern = TRUE))) {
  ids <- findNotStarted()
  ids[, chunk := chunk(job.id, chunk.size = 50)]
  submitJobs(ids = ids, # walltime in seconds, 10 days max, memory in MB
             resources = list(name = reg_name, chunks.as.arrayjobs = TRUE, 
                              ncpus = 1, memory = 6000, walltime = 10*24*3600, 
                              max.concurrent.jobs = 40))
} else {
  submitJobs()
}

waitForJobs()

# Get results -------------------------------------------------------------
reduceResultsList()
res <-  flatten(ijoin(reduceResultsDataTable(), getJobPars()))
resi = unwrap(ijoin(reduceResultsDataTable(), getJobPars()))
unlist(getJobPars(id = 50))

write.csv(apply(resi,2,as.character), "./res.csv")
