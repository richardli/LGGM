#!/bin/bash
#SBATCH --job-name Post      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short     # Slurm partition to use
#SBATCH --ntasks 1          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-05:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000     # Memory limit for each tasks (in MB)
#SBATCH -o out/Rt_%j.out    # File to which STDOUT will be written
#SBATCH -e out/Rt_%j.err    # File to which STDERR will be written

# module load R
# Rscript exp-process-3.r $SLURM_ARRAY_TASK_ID > ../experiments/log/nK3-$SLURM_ARRAY_TASK_ID

# sbatch --array=84-169 exp-process-3.sbatch

## srun --pty --partition=short --time=2:00:00 --mem-per-cpu=2500 /bin/bash
## module load R
## R
remove(list = ls())
library(data.table)
library(mnormt)
library(huge)
library(pracma)
library(mvtnorm)
library(MCMCpack)
library(truncnorm)
library(tmvtnorm)
library(matrixcalc)
library(ROCR)
library(corrplot)
source("functions.r")

evalNBprob <- function(probbase, training, testing, G, csmf = NULL, csmf.true, samepop=TRUE){
	is.testing <- 1:dim(testing)[1]
	if(samepop){
		testing <- rbind(testing, training)
	}
	Ntest <- dim(testing)[1]
	pnb.ind <- pnb.ind.inter <- pnb.ind.miss <- matrix(NA, Ntest, G)
	if(is.null(csmf)){
		csmf <- rep(1/G, G)
	}
	for(i in 1:Ntest){
		for(g in 1:G){
			pnb.ind.inter[i, g] <- prod(probbase[g, which(testing[i,-1] == "Y")], na.rm = TRUE)  * csmf[g]
			pnb.ind[i, g] <- pnb.ind.inter[i,g] * prod(1 - probbase[g, which(testing[i,-1] == "N")], na.rm = TRUE)  
			pnb.ind.miss[i, g] <- pnb.ind.inter[i,g] * prod(1 - probbase[g, which(testing[i,-1] != "Y")], na.rm = TRUE)  
		}
		if(sum(pnb.ind[i, ]) > 0) pnb.ind[i, ] <- pnb.ind[i, ] / sum(pnb.ind[i, ])
		if(sum(pnb.ind.inter[i, ]) > 0) pnb.ind.inter[i, ] <- pnb.ind.inter[i, ] / sum(pnb.ind.inter[i, ])
		if(sum(pnb.ind.miss[i, ]) > 0) pnb.ind.miss[i, ] <- pnb.ind.miss[i, ] / sum(pnb.ind.miss[i, ])
		cat(".")
	}
	
    fitted.nb <- getAccuracy(pnb.ind[is.testing,], testing[is.testing, 1], csmf=csmf.true)[1:4]
	fitted.inter <- getAccuracy(pnb.ind.inter[is.testing, ], testing[is.testing, 1], csmf=csmf.true)[1:4]
	fitted.miss <- getAccuracy(pnb.ind.miss[is.testing, ], testing[is.testing, 1], csmf=csmf.true)[1:4]
	
	csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}

	csmf.nb <- apply(pnb.ind, 2, mean)
	csmf.inter <- apply(pnb.ind.inter, 2, mean)
	csmf.miss <- apply(pnb.ind.miss, 2, mean)
	
	fitted.nb[1] <- csmfacc(csmf.nb, csmf.true)
	fitted.inter[1] <- csmfacc(csmf.inter, csmf.true)
	fitted.miss[1] <- csmfacc(csmf.miss, csmf.true)

	return(list(csmf.true = csmf.true, csmf.nb = csmf.nb, csmf.inter = csmf.inter, fitted.nb = fitted.nb, fitted.inter = fitted.inter, pnb = pnb.ind, pnb.inter = pnb.ind.inter, pnb.miss = pnb.ind.miss, fitted.miss=fitted.miss))
}
getProbbase <- function(train, G, causes){
	P <- dim(train)[2] - 2
	probbase <- matrix(NA, G, P)
	prior_sub <- train
	for(i in 1:G){
		probbase[i, ] <- apply(prior_sub[prior_sub[, 2] == causes[i], -c(1:2)], 2, function(x){sum(x=="Y") / sum(x != ".")})
	}
	mean <- apply(probbase, 2, mean, na.rm = TRUE)
	for(i in 1:P){
		probbase[is.na(probbase[, i]), i] <- mean[i]
	}
	probbase[is.na(probbase)] <- 0.5
	probbase[probbase == 0] <- min(probbase[probbase>0]) / 2
	probbase[probbase == 1] <- 1 - (1 - max(probbase[probbase<1]))/2
	return(probbase)
}
getProbbase2 <- function(train, G, causes){
	P <- dim(train)[2] - 2
	probbase <- matrix(NA, G, P)
	prior_sub <- train
	for(i in 1:G){
		probbase[i, ] <- apply(prior_sub[prior_sub[, 2] == causes[i], -c(1:2)], 2, function(x){(1+sum(x=="Y")) / (2+sum(x != "."))})
	}
	return(probbase)
}

getProbbase3 <- function(train, G, causes){
	P <- dim(train)[2] - 2
	probbase <- matrix(NA, G, P)
	prior_sub <- train
	for(i in 1:G){
		probbase[i, ] <- apply(prior_sub[prior_sub[, 2] == causes[i], -c(1:2)], 2, function(x){sum(x=="Y") / length(x)})
	}
	mean <- apply(probbase, 2, mean, na.rm = TRUE)
	for(i in 1:P){
		probbase[is.na(probbase[, i]), i] <- mean[i]
	}
	probbase[is.na(probbase)] <- 0.5
	probbase[probbase == 0] <- min(probbase[probbase>0]) / 2
	probbase[probbase == 1] <- 1 - (1 - max(probbase[probbase<1]))/2
	return(probbase)
}

index <- as.numeric(commandArgs(trailingOnly = TRUE)[1])
config <- NULL

for(pr in c(3)){
for(case in c(20)){
for(type in 0:3){
for(test in 1:20){
	informative <- "No"		
	t1 <- paste0("typeK", pr)
	post <- "F"
	hastrain <- "Yes"
	if(type == 0) hastrain <- "No"

	config <- rbind(config, 
			c(paste0(t1),  
			  paste0("K_", test, "_train", type), 
			  paste0("K_", test, "_test", type), 
			  paste0("newK9-", test,"-",type,"-",case,post), 
			  "SSSL", "Yes", hastrain, informative))
}}}}

config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-20F", "SSSL", "Yes", "Yes", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-21F", "SSSL", "Yes", "No", "No"))  
config <- rbind(config, c("typeK0", "K_all0", "K_all0", "newK0-1F", "SSSL", "Yes", "No", "No"))


for(pr in c(3)){
for(case in c("C")){
for(type in 0:3){
for(test in 1:20){
	informative <- "No"		
	t1 <- paste0("typeK", pr)
	post <- ""
	hastrain <- "Yes"
	if(type == 0) hastrain <- "No"

	config <- rbind(config, 
			c(paste0(t1),  
			  paste0("K_", test, "_train", type), 
			  paste0("K_", test, "_test", type), 
			  paste0("newK9-", test,"-",type,"-",case,post), 
			  "SSSL", "Yes", hastrain, informative))
}}}}


config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-A", "SSSL", "Yes", "Yes", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-B", "SSSL", "Yes", "No", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-C", "SSSL", "Yes", "Yes", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-D", "SSSL", "Yes", "No", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-C0", "SSSL", "Yes", "Yes", "No")) 
config <- rbind(config, c("typeK3", "K_train0", "K_test0", "newK9-0-0-D0", "SSSL", "Yes", "No", "No")) 


args_in <- config[index, ]

dir <- "../experiments/"
subdir <- "expnew/"

dir0 <- paste0(subdir, args_in[1])
dir1 <-paste0(subdir, args_in[2])
if(args_in[2] != "K_train0") args_in[3] <- gsub("test0", "test1", args_in[3])
dir2 <-paste0(subdir, args_in[3])
name <- paste0(args_in[4])
# name2 <-  gsub("-true-true", "-false-true", name)
name2 <- name
typecov <- args_in[5]
is.classification <- args_in[6] == "Yes"
has.train <- args_in[7] == "Yes"
is.informative <- args_in[8] == "Yes"
dir0delta <- dir0
# # fix for file naming issues
# name2 <- name
# if((!has.train) && index < n1212) name2 = paste0(name, "-train")
if(file.exists(paste0(dir, name, "/", name2, "_mean_out_mean.txt"))){
	print(args_in)
	print("Run finished.")
}else{
	print(args_in)
	stop("Run not finished.")
}

done <- FALSE
if(has.train){
	exist <- file.exists(paste0("rdaVA/201802/", name, ".rda"))
	if(exist){
		load(paste0("rdaVA/201802/", name, ".rda"))
		print(out$metric)
		print("Done")
		done <- TRUE
		# stop()
	}
}else{
	exist <- file.exists(paste0("rdaVA/201802/", name, "notrain.rda"))
	if(exist){
		load(paste0("rdaVA/201802/", name, "notrain.rda"))
		print(out$metric)
		print("Done")
		# stop()
	}
}
 
csmf <- read.csv(paste0("../data/expnew/csmf.csv"), header = F)
csmf <- as.numeric(as.matrix(csmf))
if(!is.informative){
	G <- length(csmf)
	csmf <- rep(1/G, G)
} 

delta <- read.csv(paste0("../data/", dir0delta, "_delta.csv"), header = F)
if(has.train){
	train_sub <- read.csv(paste0("../data/", dir1, ".csv"), header = F)
}else{
	train_sub <- NULL
}
test_sub <- read.csv(paste0("../data/", dir2, ".csv"), header = F)

probbase <- 1 - pnorm(as.matrix(delta))
G <- dim(probbase)[1]
P <- dim(probbase)[2]

causes <- c(test_sub[, 1])
# causes <- c(test_sub[, 1], train_sub[, 1])
csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))

# if(has.train){
# 	offset <- 1
# 	if(sum(1:G %in% train_sub[,1]) < G) offset <- 0
# 	csmf.train <- as.numeric((table(c(1:G, train_sub[, 1])) -offset )/(dim(train_sub)[1] + G * (offset == 0)))
# }else{
# 	csmf.train <-rep(1/G, G)
# }
csmf.train <- csmf
# csmf.train <-rep(1/G, G)
######################################
if(is.classification && args_in[3] != "K_all0"){
	fit.nb <- evalNBprob(probbase = probbase, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
	

	if(has.train){
		if(args_in[1] == "typeK2"){
			probbase2 <- getProbbase2(cbind(1,train_sub), G, 1:G)
		}
		if(args_in[1] == "typeK3"){
			probbase2 <- getProbbase3(cbind(1,train_sub), G, 1:G)
		}
		
		fit.nb.noprior <- evalNBprob(probbase = probbase2, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
		fitted.nb.noprior <- fit.nb.noprior$fitted.nb
	}else{
		fit.nb.noprior <- NULL
		fitted.nb.noprior <- NA
	}
	#####################################
	pnb <- fit.nb$pnb
	pick.nb <- apply(pnb, 1, which.max)
	print(table(pick.nb[1:dim(test_sub)[1]], test_sub[, 1]))
}else{
	fit.nb <-  fit.nb.noprior <- pick.nb <- pick.draw <- pick.int <- prob.mean <- pnb_integral <- NULL
	fitted.nb.noprior  <- NA
}
print("Naive Bayes")
print(fit.nb$fitted.nb)
print(fit.nb.noprior$fitted.nb)
######################################
prefix <- paste0(dir, name, "/", name2)
prec.fit <- as.matrix(fread(paste0(prefix, "_invcorr_out_mean.txt"), sep = ","))
mean.fit <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))
corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out_mean.txt"), sep = ","))
inclusion.fit <- as.matrix(fread(paste0(prefix, "_inclusion_out.txt"), sep = ","))
inclusion.s1 <- as.matrix(fread(paste0(prefix, "_s1_inclusion_out.txt"), sep = ","))
inclusion.fit <- inclusion.fit - inclusion.s1


if(dim(prec.fit)[1] != dim(delta)[2]){
	stop("Precision matrix dimension wrong, check runscript")
}

if(is.classification){
	prob.fit <- as.matrix(fread(paste0(prefix, "_prob_out.txt"), sep = ","))[, -1]
	assignment.fit <- as.matrix(fread(paste0(prefix, "_assignment_out_mean.txt"), sep = ","))
	Nitr <- dim(prob.fit)[2]

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
	if(dim(prob.fit)[1] != dim(delta)[1]){
		stop("Fitted CSMF dimension wrong, check runscript")
	}

}else{
	prob.fit <- NULL
	assignment.fit <- NULL
	Nitr <- NA
}

type <- rep(1, P)
# corr.mean <- prec.mean <- cov.mean <- matrix(0, P, P)
# delta.mean <- matrix(0, G, P)
# delta.fit <- array(0, dim = c(Nitr, G, P))
# for(i in 2:(Nitr+1)){
# 	corr.mean <- corr.mean + corr.fit[((i-1) * P + 1):(i * P), ]
# 	prec.mean <- prec.mean + prec.fit[((i-1) * P + 1):(i * P), ]
# 	cov.mean <- cov.mean + solve(prec.fit[((i-1) * P + 1):(i * P), ])
# 	delta.mean <- delta.mean + mean.fit[((i-1) * G + 1):(i * G), ]
# 	delta.fit[i-1, , ] <- mean.fit[((i-1) * G + 1):(i * G), ]
# }
corr.mean <- corr.fit  
prec.mean <- prec.fit 
# cov.mean <- cov.mean / Nitr
delta.mean <- mean.fit
if(is.classification){
	Ntest <- dim(test_sub)[1] 
	test0 <- matrix(NA, dim(test_sub)[1], P)
	test0[test_sub[,-1] == "Y"] <- 1
	test0[test_sub[,-1] == "N"] <- 0
	counter <- 1
	prob.mean <- t(assignment.fit)
	prob.fit.summary <- apply(prob.fit, 1, function(x){c(mean(x), quantile(x, c(.025, .5, .975)))})

	csmf.mean <- prob.fit.summary[1, ]
	fitted <- getAccuracy(prob.mean, test_sub[, 1], csmf.true, csmf.mean)[1:4]
	if(!is.null(assignment.fit.s1)){
		prob.mean.s1 <- t(assignment.fit.s1)
		prob.fit.summary.s1 <- apply(prob.fit.s1, 1, function(x){c(mean(x), quantile(x, c(.025, .5, .975)))})
		csmf.mean.s1 <- prob.fit.summary.s1[1, ]
		fitted.s1 <- getAccuracy(prob.mean.s1, test_sub[, 1], csmf.true, csmf.mean.s1)[1:4]
	}

	tmp <- rbind(
		  fitted.s1,
		  rep(NA, 4),
		  fitted, 
		  rep(NA, 4),
		  fit.nb$fitted.nb,
		  fitted.nb.noprior,
		  fit.nb$fitted.inter)
		  # fit.nb.noprior$fitted.inter)
	if(args_in[3] != "K_all0"){
		colnames(tmp) <- c("CSMF", "Top1", "Top2", "Top3")
		rownames(tmp) <- c("Fitted-S1", "Integral-S1", "Fitted", "Integral", "NaiveBayes", "NaiveBayes_train","InterVA")
		print(tmp)
	}
	pick.draw <- apply(prob.mean, 1, which.max)
	print(table(pick.draw, test_sub[, 1]))

	
	out <- list(corr.mean = corr.mean, 
				prec.mean = prec.mean, 
				delta.mean = delta.mean, 
				corr.mean.s1 = corr.fit.s1, 
				prec.mean.s1 = prec.fit.s1, 
				delta.mean.s1 = mean.fit.s1, 
				inclusion = inclusion.s1,
				metric = tmp, 
				truth = test_sub[, 1], 
				pick.nb = pick.nb, 
				pick.draw = pick.draw, 
				csmf = prob.fit, 
				csmf.s1 = prob.fit.s1, 
				assignment.fit.s1 = assignment.fit.s1,
				assignment.fit = assignment.fit,
				fitted.nb = fit.nb, 
				fitted.nb.noprior = fit.nb.noprior)
	if(has.train){
		save(out, file = paste0("rdaVA/201802/", name, ".rda"))
	}else{
		save(out, file = paste0("rdaVA/201802/", name, "notrain.rda"))
	}

	csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}


	# first stage
	pnb_integral.s1 <- plug_in_estimator(test0, type, corr.fit.s1, mean.fit.s1, csmf.mean.s1)
	pick.int.s1 <- apply(pnb_integral.s1, 1, which.max)
	print(table(pick.int.s1, test_sub[, 1]))
	fitted_integral.s1 <- getAccuracy(pnb_integral.s1, test_sub[, 1], csmf.true, csmf.mean.s1)[1:4]
	fitted_integral.s1[1] <- csmfacc(csmf.true, apply(pnb_integral.s1, 2, mean))

	# post selection
	pnb_integral <- plug_in_estimator(test0, type, corr.mean, delta.mean, csmf.mean)
	pick.int <- apply(pnb_integral, 1, which.max)
	print(table(pick.int, test_sub[, 1]))
	fitted_integral <- getAccuracy(pnb_integral, test_sub[, 1], csmf.true, csmf.mean)[1:4]
	fitted_integral[1] <- csmfacc(csmf.true, apply(pnb_integral, 2, mean))

	tmp[2, ] <- fitted_integral.s1
	tmp[4, ] <- fitted_integral
	print(fitted_integral)
}else{
	tmp <- NULL
}

# update results
out$pnb_integral <- pnb_integral
out$pnb_integral.s1 <- pnb_integral.s1
out$pick.int <- pick.int
out$pick.int.s1 <- pick.int.s1
out$metric <- tmp
if(has.train){
	save(out, file = paste0("rdaVA/201802/", name, ".rda"))
}else{
	save(out, file = paste0("rdaVA/201802/", name, "notrain.rda"))
}

