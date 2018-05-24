## This script calculates the related output for one LatentModel fit
## i.e., no classification.
##

remove(list = ls())
# install.packages(c("mnormt", "huge","pracma", "mvtnorm", "MCMCpack", "truncnorm", "tmvtnorm", "matrixcalc", "ROCR"))
library("data.table")
library("mnormt")
library("huge")
library("pracma")
library("mvtnorm")
library("MCMCpack")
library("truncnorm")
library("tmvtnorm")
library("matrixcalc")
library("ROCR")
source("functions.r")

# THIS arguments are followed by the experiment in README.md
dir <- "../data/"
G <- 1
N <- 200
P <- 50
compact <- TRUE
pre <- "test1"
miss.rate <- "0.2"
typecov <- "SSSL"
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
		tmp <- rocf1(prec.mean, as.matrix(s0))
		auc.thres <-  tmp$AUC
		f1.thres <-  max(tmp$F1)
	}else{
		pred.sssl <- NULL
		auc.sssl <- f1.sssl <- NULL
		auc.thres <- f1.thres <-  NULL
	}

	allout[count, , 1:4] <- norm
	allout[count, , 5:8] <- norm2
	for(i in 1: length(aucs)){
		allout[count, i, 9:10] <- c(aucs[[i]]$AUC, max(aucs[[i]]$F1))
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
	save(allout, file = paste0("../data/processed/",typecov, pre, miss.rate, "metrics1.rda"))			
}

