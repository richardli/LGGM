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
	
	csmfacc <- function(csmf.fit, csmf){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}

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

getProbbase3 <- function(train, G, causes, minval=NULL, maxval=NULL){
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
	if(is.null(minval)){
		minval <- min(probbase[probbase>0]) / 2
	}
	if(is.null(maxval)){
		maxval <- 1 - (1 - max(probbase[probbase<1]))/2
	}
	probbase[probbase == 0] <- minval
	probbase[probbase == 1] <- maxval
	return(probbase)
}

rep <- 1
has.train <- TRUE
compute_integral <- FALSE # set this to TRUE to reproduce results in the paper
args_in <- c(paste0("typePS", rep), 
			 paste0("PS_0_train", rep), 
			 paste0("PS_0_test", rep), 
			 paste0("rep", rep), "SSSL")

dir <- "../data/"
subdir <- ""

dir0 <- paste0(subdir, args_in[1])
dir1 <-paste0(subdir, args_in[2])
dir2 <-paste0(subdir, args_in[3])
name <- paste0(args_in[4])
typecov <- args_in[5]
is.classification <- TRUE
is.informative <- FALSE
dir0delta <- dir0
if(file.exists(paste0(dir, name, "/", name, "_mean_out_mean.txt"))){
	print(args_in)
	print("Run finished.")
}else{
	print(args_in)
	stop("Run not finished.")
}

delta <- read.csv(paste0("../data/phmrc/", dir0delta, "_delta.csv"), header = F)
G <- dim(delta)[1]
csmf <- rep(1/G, G)
csmf.train <- csmf

if(has.train){
	train_sub <- read.csv(paste0("../data/phmrc/", dir1, ".csv"), header = F)
}else{
	train_sub <- NULL
}
test_sub <- read.csv(paste0("../data/phmrc/", dir2, ".csv"), header = F)

probbase <- 1 - pnorm(as.matrix(delta))
G <- dim(probbase)[1]
P <- dim(probbase)[2]

causes <- c(test_sub[, 1])
# causes <- c(test_sub[, 1], train_sub[, 1])
csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))

# csmf.train <-rep(1/G, G)
######################################
if(is.classification){
	fit.nb <- evalNBprob(probbase = probbase, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
	fit.nb.noprior <- NULL
	fitted.nb.noprior <- NA
	#####################################
	pnb <- fit.nb$pnb
	pick.nb <- apply(pnb, 1, which.max)
	# print(table(pick.nb[1:dim(test_sub)[1]], test_sub[, 1]))
}else{
	fit.nb <-  fit.nb.noprior <- pick.nb <- pick.draw <- pick.int <- prob.mean <- pnb_integral <- NULL
	fitted.nb.noprior  <- NA
}

prefix <- paste0(dir, name, "/", name)
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
corr.mean <- corr.fit  
prec.mean <- prec.fit 
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
		  # fitted, 
		  # rep(NA, 4),
		  fit.nb$fitted.nb,
		  fit.nb$fitted.inter)
		  # fit.nb.noprior$fitted.inter)
	if(args_in[3] != "K_all0"){
		colnames(tmp) <- c("CSMF", "Top1", "Top2", "Top3")
		rownames(tmp) <- c("Fitted", "Integral", "NaiveBayes", "InterVA")
		print(tmp)
	}
	pick.draw <- apply(prob.mean, 1, which.max)
	# print(table(pick.draw, test_sub[, 1]))

	
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
	csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}


	# first stage
	if(compute_integral){
		pnb_integral.s1 <- plug_in_estimator(test0, type, corr.fit.s1, mean.fit.s1, csmf.mean.s1)
		pick.int.s1 <- apply(pnb_integral.s1, 1, which.max)
		# print(table(pick.int.s1, test_sub[, 1]))
		fitted_integral.s1 <- getAccuracy(pnb_integral.s1, test_sub[, 1], csmf.true, csmf.mean.s1)[1:4]
		fitted_integral.s1[1] <- csmfacc(csmf.true, apply(pnb_integral.s1, 2, mean))
		# update results
		out$pnb_integral.s1 <- pnb_integral.s1
		out$pick.int.s1 <- pick.int.s1
		out$metric[2, ] <- fitted_integral.s1
	}
}else{
	tmp <- NULL
}

if(has.train){
	save(out, file = paste0("../data/processed/", name, ".rda"))
}else{
	save(out, file = paste0("rdaVA/processed/", name, "notrain.rda"))
}
 