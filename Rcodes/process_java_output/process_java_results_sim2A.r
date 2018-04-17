#!/bin/bash
#SBATCH --job-name c2      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium     # Slurm partition to use
#SBATCH --ntasks 1          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 2-00:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000     # Memory limit for each tasks (in MB)
#SBATCH -o out/Rt_%j.out    # File to which STDOUT will be written
#SBATCH -e out/Rt_%j.err    # File to which STDERR will be written

# module load R
# Rscript process_java_results2.r $SLURM_ARRAY_TASK_ID > ../experiments/log/class-$SLURM_ARRAY_TASK_ID
# sbatch --array=1-180 process_java_results2.sbatch


## srun --pty --partition=short --time=2:00:00 --mem-per-cpu=2500 /bin/bash


remove(list = ls())
library(data.table)
library("mnormt")
library("huge")
library("pracma")
library("mvtnorm")
library("MCMCpack")
library("truncnorm")
library("tmvtnorm")
library("matrixcalc")
library("ROCR")
source("FanEstimator.r")

index <- as.numeric(commandArgs(trailingOnly = TRUE)[1])
compact <- TRUE
config <- NULL
types <- c("CC", "DD")
covs <- c("Random")
Ns <- c(0, 80, 200)
misses <- c("0.0","0.2", "0.5")
methods <- c("SSSL")
allseeds <- seq(1, 46, by = 5)
config <- expand.grid(types, covs, Ns, misses, methods, allseeds)
config[] <- lapply(config, as.character)
args_in <- as.character(config[index, ])

dir <- "../experiments/new"
G <- 20
Ntest <- 800
P <- 50
name <- args_in[1]
type <- args_in[2]
N <- as.numeric(args_in[3]) + Ntest
miss <- args_in[4]
method <- args_in[5]
args <- c(paste0(name, type, N-Ntest), method, miss)
start <- as.numeric(args_in[6])


# args <- c("D1A0", "PX", "0.5")
pre <- args[1]
typecov <- args[2]
miss.rate <- args[3]
prob <- TRUE
fixrep <- 0
subdir <- paste0(pre, typecov, "N", N, "P", P, "Miss", miss.rate)


files <- list.files(paste0(dir, subdir, "/"))
tmp <- gsub("Rep", "!", files)
if(compact){
	tmp <- gsub("_corr_out_mean.txt", "!", tmp)
}else{
	tmp <- gsub("_corr_out.txt", "!", tmp)
}
tmp <- unlist(strsplit(tmp, "!"))
# force the non-rep number to be NA
seeds <- sort(unique(as.numeric(tmp)))
print(paste0("Total number of fit: ", length(seeds)))


seeds <- seeds[start : (start + 4)]
metric2 <- array(NA, dim = c(5, 6, 4))
index.out <- which(allseeds == start)

for(kk in 1:5){

	rep <- seeds[kk]
	cat(".\n")
	prefix <- paste0(dir, subdir, "/new", subdir, "Rep", rep)
	
	if(compact){
		if(file.exists(paste0(prefix, "_corr_out_core_mean.txt"))){
			corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
			corr.fit2 <- as.matrix(fread(paste0(prefix, "_corr_out_mean.txt"), sep = ","))
		}else{
			corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out_mean.txt"), sep = ","))
			corr.fit2 <- corr.fit
		}
		if(typecov == "SSSL"){
			inclusion.fit <- as.matrix(fread(paste0(prefix, "_inclusion_out.txt"), sep = ","))			
		}else{
			inclusion.fit <- matrix(1, P, P)
		}
		prec.fit <- as.matrix(fread(paste0(prefix, "_invcorr_out_mean.txt"), sep = ","))
		mean.fit <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))
		assignment.fit <- as.matrix(fread(paste0(prefix, "_assignment_out_mean.txt"), sep = ","))
		if(file.exists(paste0(prefix, "_s1_assignment_out_mean.txt"))){
			prec.fit.s1 <- as.matrix(fread(paste0(prefix, "_s1_invcorr_out_mean.txt"), sep = ","))
			mean.fit.s1 <- as.matrix(fread(paste0(prefix, "_s1_mean_out_mean.txt"), sep = ","))
			corr.fit.s1 <- as.matrix(fread(paste0(prefix, "_s1_corr_out_mean.txt"), sep = ","))
			assignment.fit.s1 <- as.matrix(fread(paste0(prefix, "_s1_assignment_out_mean.txt"), sep = ","))
			prob.fit.s1 <- as.matrix(fread(paste0(prefix, "_s1_prob_out.txt"), sep = ","))[, -1]
		}else{
			assignment.fit.s1 <- NULL
			prob.fit.s1 <- NULL
		}
	}else{
		corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out.txt"), sep = ","))
		if(typecov == "SSSL"){
		inclusion.fit <- as.matrix(fread(paste0(prefix, "_inclusion_out.txt"), sep = ","))					
		}else{
			inclusion.fit <- matrix(1, P, P)
		}
		# prec.fit <- as.matrix(fread(paste0(prefix, "_prec_out.txt"), sep = ","))
		mean.fit <- as.matrix(fread(paste0(prefix, "_mean_out.txt"), sep = ","))
		Nitr <- dim(corr.fit)[1] / P - 1	
		assignment.fit <- as.matrix(fread(paste0(prefix, "_assignment_out.txt"), sep = ","))
	}
	corr.true <- as.matrix(fread(paste0(prefix, "_corr_out_truth.txt"), sep = ","))
	prec.true <- as.matrix(fread(paste0(prefix, "_prec_out_truth.txt"), sep = ","))
	mean.true <- as.matrix(fread(paste0(prefix, "_mean_out_truth.txt"), sep = ","))
	X <- as.matrix(fread(paste0(prefix, "_X.txt"), sep = ","))
	
	
	prob.true <- as.vector(fread(paste0(prefix, "_prob_true.txt"), sep = ","))
	prob.fit <- as.matrix(fread(paste0(prefix, "_prob_out.txt"), sep = ","))[, -1]
	Nitr <- dim(prob.fit)[2]


	## handle -Double.Max from java
	X[X < -1e10] <- NA

	type <- rep(1, dim(X)[2] - 1)
	for(i in 2:dim(X)[2]){
		## todo: handle categorical case
		if(sum(!(X[,i] %in% c(0, 1, NA)), na.rm=T)>0){
			type[i-1] <- 0
		}
	}


	membership <- X[,1] + 1
	corr.true <- cov2cor(solve(prec.true))
	corr.mean <- prec.mean <- cov.mean <- matrix(0, P, P)
	if(compact){
		corr.mean <- corr.fit
		prec.mean <- prec.fit
		delta.mean <- mean.fit
	}else{
		delta.mean <- matrix(0, G, P)
		delta.fit <- array(0, dim = c(Nitr, G, P))
 		for(i in 2:(Nitr+1)){
			corr.mean <- corr.mean + corr.fit[((i-1) * P + 1):(i * P), ]
			# prec.mean <- prec.mean + prec.fit[((i-1) * P + 1):(i * P), ]
			# cov.mean <- cov.mean + solve(prec.fit[((i-1) * P + 1):(i * P), ])
			delta.mean <- delta.mean + mean.fit[((i-1) * G + 1):(i * G), ]
			delta.fit[i-1, , ] <- mean.fit[((i-1) * G + 1):(i * G), ]
		}
		corr.mean <- corr.mean / Nitr
		prec.mean <- prec.mean / Nitr
		cov.mean <- cov.mean / Nitr
		delta.mean <- delta.mean / Nitr
	}
	

	Ntest <- dim(assignment.fit)[2]  
	train <- X[-c(1:Ntest), ]
	test <- X[c(1:Ntest), ]
	membership.train <- membership[-c(1:Ntest)]
	if(compact){
		prob.mean <- t(assignment.fit)
	}else{
		counter <- 1
		prob.ind <- array(0, dim = c(Nitr, Ntest, G))
		for(g in 1:G){
			counter <- counter + 1 # remove the extra 0 line from Java
			for(itr in 1:Nitr){
				prob.ind[itr, , g] <- assignment.fit[counter, ]
				counter <- counter + 1	
			}
		}
		prob.mean <- apply(prob.ind, c(2,3), mean, na.rm = TRUE)
		population_mean <- apply(prob.ind, 3, mean, na.rm = TRUE)
	}
	prob.fit.summary <- apply(prob.fit, 1, function(x){c(mean(x), quantile(x, c(.025, .5, .975)))})

	fitted_nb <- naivebayes_eval(type = type, 
					train = X[-(1:Ntest), -1], 
					test = X[1:Ntest, -1], 
					train.Y = membership[-(1:Ntest)], 
					test.Y = membership[1:Ntest], 
					csmf = prob.true,  
					mean.true = -mean.true)


	csmf.true <- as.numeric(prob.true)
	csmf.mean <- prob.fit.summary[1, ]
	fitted <- getAccuracy(prob.mean, membership[1:Ntest], csmf.true, csmf.mean)[1:4]

	if(!is.null(assignment.fit.s1)){
		prob.mean.s1 <- t(assignment.fit.s1)
		prob.fit.summary.s1 <- apply(prob.fit.s1, 1, function(x){c(mean(x), quantile(x, c(.025, .5, .975)))})
		csmf.mean.s1 <- prob.fit.summary.s1[1, ]
		fitted.s1 <- getAccuracy(prob.mean.s1, membership[1:Ntest], csmf.true, csmf.mean.s1)[1:4]
	}

	print(fitted_nb[, 1:4])
	print(fitted)
	print(fitted.s1)
	# print(rbind(as.numeric(prob.true), prob.fit.summary ))
	# print(get4norm(list(Fan.unknown.combine, corr.mean), corr.true))


	csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}
	pnb_integral <- plug_in_estimator(test[1:Ntest, -1], type, corr.fit2, delta.mean, csmf.mean)
	fitted_integral <- getAccuracy(pnb_integral, membership[1:Ntest], csmf.true, NULL)[1:4]


	pnb_integral.s1 <- plug_in_estimator(test[1:Ntest, -1], type, corr.fit.s1, mean.fit.s1, csmf.mean.s1)
	fitted_integral.s1 <- getAccuracy(pnb_integral.s1, membership[1:Ntest], csmf.true, NULL)[1:4]


	fitted_Fan <- rep(NA, 4)
	tmp <- rbind(fitted.s1,
				fitted_integral.s1, 
				fitted,
				fitted_integral, 
				fitted_nb[, 1:4]
		)
	
	metric2[kk, , ] <- tmp
	
	dimnames(metric2)[[2]] <- c("FittedS1", "Plug-inS1","Fitted", "Plug-in",  "NaiveBayes", "InterVA")
	dimnames(metric2)[[3]] <- c("CSMF", "top1", "top2", "top3")
	print(apply(metric2, c(2, 3), mean, na.rm = TRUE))

	save(metric2, file = paste0("rdaVA/201802/sim/", pre, typecov, miss.rate, "-", index.out, "prediction-0309.rda"))
	print(kk)
}


