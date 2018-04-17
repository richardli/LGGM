#!/bin/bash
#SBATCH --job-name j0     # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium     # Slurm partition to use
#SBATCH --ntasks 1          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 2-00:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000     # Memory limit for each tasks (in MB)
#SBATCH -o out/Rt_%j.out    # File to which STDOUT will be written
#SBATCH -e out/Rt_%j.err    # File to which STDERR will be written
# module load R
# Rscript process_java_results.r $SLURM_ARRAY_TASK_ID > ../experiments/log/sim0-$SLURM_ARRAY_TASK_ID
# sbatch --array=1-12 process_java_results.sbatch

remove(list = ls())
# install.packages(c("mnormt", "huge","pracma", "mvtnorm", "MCMCpack", "truncnorm", "tmvtnorm", "matrixcalc", "ROCR"))
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

dir <- "../experiments/"
# typecov <- "PX"
# typecov <- "SSSL"
G <- 1

N <- 200
P <- 50

index <- as.numeric(commandArgs(trailingOnly = TRUE)[1])
compact <- TRUE
config <- NULL
types <- c("Case3", "Case4")
misses <- c("0.0", "0.2", "0.5")
methods <- c("SSSL", "PX")
config <- expand.grid(types, misses, methods)
config[] <- lapply(config, as.character)
args <- as.character(config[index, ])
pre <- args[1]
miss.rate <- args[2]
typecov <- args[3]


totalEdge <- 0
processed <- 0

subdir <- paste0(pre, typecov, "N", N, "P", P, "Miss", miss.rate)
files <- list.files(paste0(dir, subdir, "/"))
tmp <- gsub("Rep", "!", files)
tmp <- gsub("_corr_out_mean.txt", "!", tmp)
tmp <- unlist(strsplit(tmp, "!"))
# force the non-rep number to be NA
seeds <- sort(unique(as.numeric(tmp)))
print(paste0("Total number of fit: ", length(seeds)))

count <- 1
nrep <- min(100, length(seeds))
allout <- array(NA, c(nrep, 4, 10))

for(rep in seeds[1:nrep]){
	cat(".\n")
	# 
	# Again this script is created for the java codes containing
	#   typo of mixing corr and prec 
	#
	prefix <- paste0(dir, subdir, "/", subdir, "Rep", rep)
	if(file.exists(paste0(prefix, "_corr_out_mean.txt"))){
		corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
		corr.fit2 <- as.matrix(fread(paste0(prefix, "_corr_out_mean.txt"), sep = ","))
		if(dim(corr.fit)[1] != dim(corr.fit2)[1]){
			corr.fit = cov2cor(corr.fit2)
		}
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
	corr.true <- as.matrix(fread(paste0(prefix, "_corr_out_truth.txt"), sep = ","))
	prec.true <- as.matrix(fread(paste0(prefix, "_prec_out_truth.txt"), sep = ","))
	mean.true <- as.matrix(fread(paste0(prefix, "_mean_out_truth.txt"), sep = ","))
	X <- as.matrix(fread(paste0(prefix, "_X.txt"), sep = ","))
	
	
	# handle -Double.Max from java
	X[X < -1e10] <- NA
	
	type <- rep(1, dim(X)[2] - 1)
	for(i in 2:dim(X)[2]){
		## todo: handle categorical case
		if(sum(!(X[,i] %in% c(0, 1, NA)), na.rm=T)>0){
			type[i-1] <- 0
		}
	}


	membership <- X[,1] + 1
	corr.mean <- corr.fit
	prec.mean <- prec.fit
	cov.mean <- NULL
	delta.mean <- mean.fit
	prec.mean.hat <- prec.mean
	Fan.known <- as.matrix(nearPD(
		getFanEstimatorMix(
			X = X[, -1], 
			delta = mean.true[1, ], 
			type = type)
		, corr = TRUE, keepDiag = TRUE, maxit = 1000)$mat)
	Fan.unknown <- as.matrix(nearPD(
		getFanEstimatorMix(
			X = X[, -1], 
			delta = NULL, 
			type = type)
		, corr = TRUE, keepDiag = TRUE)$mat)	
	

	corr.mean.hat <- cov2cor(solve(prec.mean.hat))

	alllist <- list(Fan.unknown, Fan.known, 
			   		corr.mean, corr.mean.hat)
	names(alllist) <- c("Fan w/o prior","Fan w/i prior", "Bayesian (ave corr)", "Bayesian (ave prec)")

	
	alllist2 <- lapply(alllist, function(x){
		solve(x)
		})
	alllist2[["Bayesian (ave prec)"]] <- prec.mean.hat
	# alllist2[[length(alllist2)]] <- cov2cor(prec.mean)
	names(alllist2) <- names(alllist)

	norm <- get4norm(alllist, corr.true, TRUE)
	norm2 <- get4norm(alllist2, prec.true, TRUE)
	rownames(norm) <- rownames(norm2) <-names(alllist)
	out <- lapply(alllist, function(x){
				huge(x, method = "glasso", nlambda = 100,  lambda.min.ratio = 0.001)
				})
	names(out) <- names(alllist)
	fprlist <- seq(0, 1, len = 200)
	s0 <- prec.true 
	s0[s0!=0] <- 1
	diag(s0) <- 0
	s0 <- as(s0, "sparseMatrix")   
	nEdge <- sum(s0) / 2 
	aucs <- lapply(out, function(x, s0){
				huge.roc(x$path, s0)
				}, s0)
	if(typecov == "SSSL"){
		tmp <- rocf1(inclusion.fit, as.matrix(s0))
		auc.sssl <-  tmp$AUC
		f1.sssl <-  max(tmp$F1)
		tmp <- rocf1(prec.mean, as.matrix(s0))
		auc.thres <-  tmp$AUC
		f1.thres <-  max(tmp$F1)
	}else{
		pred.sssl <- NULL
		auc.sssl <- f1.sssl <- NULL
		auc.thres <- f1.thres <-  NULL
	}

	# # HBIC selection
	# prec_hat <- vector("list", length(out))
	# for(i in 1:length(out)){
	# 	hbic <- HBIC(R=alllist[[i]], icov = out[[i]]$icov, n = N, d = P)
	# 	prec_hat[[i]] <- out[[i]]$icov[[which.min(hbic)]]
	# }
	# norm3 <- get4norm(prec_hat, (prec.true), TRUE)

	# if(typecov == "SSSL"){
	# 	# add also HBIC selected cov and prec
	# 	hbic_thres <- HBIC_thre(icov = prec.mean, thre = inclusion.fit, n = N, d = P)
	# 	cutoff <- hbic_thres[[2]][which.min(hbic_thres[[1]])]
	# 	prec.mean.hbic <- prec.mean * (inclusion.fit > cutoff)
	# 	prec.mean.inverse.hbic <- solve(prec.mean.hbic)
	# 	norm <- rbind(norm, get4norm(list(prec.mean.inverse.hbic), corr.true, TRUE))
	# 	norm2 <- rbind(norm2, get4norm(list(prec.mean.hbic), prec.true, TRUE))
		
	# }else{
	# 	norm <- rbind(norm, NA)
	# 	norm2 <- rbind(norm2, NA)
	# }

	# rownames(norm)[dim(norm)[1]] <- "Bayesian (HBIC)"
	# rownames(norm2)[dim(norm2)[1]] <- "Bayesian (HBIC)"
	
	
	allout[count, , 1:4] <- norm
	allout[count, , 5:8] <- norm2
	for(i in 1: length(aucs)){
		allout[count, i, 9:10] <- c(aucs[[i]]$AUC, max(aucs[[i]]$F1))
	}
	if(typecov == "SSSL"){
		allout[count, length(aucs)-1, 9:10] <- c(auc.sssl, f1.sssl)
	}

	allout[count, length(aucs), 9:10] <- c(auc.thres, max(f1.thres, na.rm=T))
	
	dimnames(allout)[[2]] <- c(names(alllist))
	dimnames(allout)[[3]] <- c(colnames(norm), paste0("Prec_", colnames(norm)), "AUC", "max F1")

	print(paste0(subdir, " ------- case ", count))
	print(round(allout[count, , ],2))
	
	count <- count + 1	
	totalEdge <- totalEdge + nEdge
	processed <- processed + 1
	print(paste0("Total number of sims processed: ", processed))
	print(paste0("Avg number of edges: ", totalEdge/processed))
	save(allout, file = paste0("rdaVA/201802/sim/",typecov, pre, miss.rate, "metrics-0309.rda"))			
}

