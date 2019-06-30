library(BDgraph)
library(data.table)
library(mnormt)
library(huge)
library(mvtnorm)
library(MCMCpack)
library(truncnorm)
library(tmvtnorm)
library(matrixcalc)
library(ROCR)
source("functions.r")

for(Case in c(3, 4, 1, 2)){
	for(Cov in c("SSSL", "PX")){
		for(miss in c("0.0", "0.2", "0.5")){
			n <- 200
			p <- 50
			pre0 <- paste0("Case", Case, Cov, "N", n, "P", p, "Miss", miss)
			files <- list.files(paste0("../experiments/", pre0, "/"))
			files <- files[grep("X.txt", files)]
			reps <- gsub("_X.txt", "", gsub(paste0(pre0, "Rep"), "", files))
			allout <- array(NA, c(length(reps), 6, 10))
			if(file.exists(paste0("../data/processed/", pre0, "-metrics1.rda"))) next

			for(ii in 1:length(reps)){
				rep <- reps[ii]
				prefix <- paste0("../experiments/", pre0, "/", pre0, "Rep", rep)

				X <- as.matrix(fread(paste0(prefix, "_X.txt")))
				prec.fit <- as.matrix(fread(paste0(prefix, "_invcorr_out_mean.txt"), sep = ","))
				corr.fit <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
				# corr.fit2 <- as.matrix(fread(paste0(prefix, "_corr_out_mean.txt"), sep = ","))
				# if(dim(corr.fit)[1] != dim(corr.fit2)[1]){
				# 	corr.fit = cov2cor(corr.fit2)
				# }	
				mean.fit <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))
				corr.true <- as.matrix(fread(paste0(prefix, "_corr_out_truth.txt"), sep = ","))
				prec.true <- as.matrix(fread(paste0(prefix, "_prec_out_truth.txt"), sep = ","))
				mean.true <- as.matrix(fread(paste0(prefix, "_mean_out_truth.txt"), sep = ","))

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
				corr.mean.hat <- cov2cor(solve(prec.mean.hat))

				if(Cov == "PX"){			
					# List of  R
					alllist <- list(corr.mean, corr.mean.hat)
					# List of  R inverse
					alllist2 <- list(solve(corr.mean), prec.mean.hat)
					# Norms of bias
					norm <- get4norm(alllist, corr.true, TRUE)
					norm2 <- get4norm(alllist2, prec.true, TRUE)
					allout[ii, 3:4, 1:4] <- norm
					allout[ii, 3:4, 5:8] <- norm2
					save(allout, file = paste0("../data/processed/", pre0, "-metrics1.rda"))	
					next
				}

				inclusion.fit <- as.matrix(fread(paste0(prefix, "_inclusion_out.txt"), sep = ","))	

				###############################################
				#### Rank based
				###############################################
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


				###############################################
				#### RJ-MCMC
				###############################################
				fbd <- bdgraph(X[, -1], method = "gcgm", algorithm = c("bdmcmc", "rjmcmc")[2], iter = 10000, burnin = 5000)
				g.rj <- as.matrix(fbd$p_links) 
				g.rj <- g.rj + t(g.rj)
				cor.rj <- cov2cor(solve(fbd$K_hat))

				###############################################
				#### Birth-death
				###############################################
				fbd1 <- bdgraph(X[, -1], method = "gcgm", algorithm = c("bdmcmc", "rjmcmc")[1], iter = 10000, burnin = 5000)
				g.bd <- as.matrix(fbd1$p_links) 
				g.bd <- g.bd + t(g.bd)
				cor.bd <- cov2cor(solve(fbd1$K_hat))

				###############################################
				#### Metrics
				###############################################
				# List of all R
				alllist <- list(Fan.unknown, Fan.known, 
						   		corr.mean, corr.mean.hat, 
						   		cor.rj, cor.bd)
				names(alllist) <- c("Fan w/o prior","Fan w/i prior", "Bayesian (ave corr)", "Bayesian (ave prec)", "G-Wishart RJ", "G-Wishart BD")

				# List of all R inverse
				alllist2 <- list(solve(Fan.unknown), solve(Fan.known), 
						   		solve(corr.mean), prec.mean.hat, 
						   		solve(cor.rj), solve(cor.bd))
				names(alllist2) <- names(alllist)

				# Norms of bias
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
				roc <- NULL
				roc[[1]] <- huge.roc(out[[1]]$path, s0)
				roc[[2]] <- huge.roc(out[[2]]$path, s0)
				diag(inclusion.fit) <- 0
				roc[[3]] <- rocf1(inclusion.fit, as.matrix(s0))
				tmp <- abs(prec.mean)
				diag(tmp) <- 0
				roc[[4]] <- rocf1(tmp, as.matrix(s0))
				roc[[5]] <- rocf1(g.rj, as.matrix(s0))
				roc[[6]] <- rocf1(g.bd, as.matrix(s0))

				allout[ii, , 1:4] <- norm
				allout[ii, , 5:8] <- norm2
				for(i in 1:6) allout[ii, i, 9:10]<- c(roc[[i]]$AUC, max(roc[[i]]$F1, na.rm=T))
				
				dimnames(allout)[[2]] <- c(names(alllist))
				dimnames(allout)[[3]] <- c(colnames(norm), paste0("Prec_", colnames(norm)), "AUC", "max F1")

				print(paste0(pre0, " ------- case ", ii))
				print(round(allout[ii, , ],2))

				# # totalEdge <- totalEdge + nEdge
				# processed <- processed + 1
				# print(paste0("Total number of sims processed: ", processed))
				# print(paste0("Avg number of edges: ", totalEdge/processed))
				save(allout, file = paste0("../data/processed/", pre0, "-metrics1.rda"))	
			}
		}
	}
}
 

all <- NULL
name <- rep(c("correct", "misspecified"), 2)
for(Case in c(3, 4, 1, 2)){
	for(miss in c("0.0", "0.2", "0.5")){
		n <- 200
		p <- 50
		Cov <- "PX"
		pre0 <- paste0("Case", Case, Cov, "N", n, "P", p, "Miss", miss)
		load(paste0("../data/processed/", pre0, "-metrics1.rda"))
		tmp0 <- apply(allout, c(2, 3), mean, na.rm = TRUE)[3:4, ]
		Cov <- "SSSL"
		pre0 <- paste0("Case", Case, Cov, "N", n, "P", p, "Miss", miss)
		load(paste0("../data/processed/", pre0, "-metrics1.rda"))
		tmp1 <- apply(allout, c(2, 3), mean, na.rm = TRUE)
		# tmp1[3, 9:10] <- tmp1[4, 9:10]
		tmp <- rbind(tmp1[c("Fan w/i prior"), ], tmp0[1, ], tmp1[c( "G-Wishart RJ", "G-Wishart BD", "Bayesian (ave corr)"), ])
		rownames(tmp) <- c("Semi-parametric", "Uniform prior", "G-Wishart RJ", "G-Wishart BD", "Spike-and-Slab prior")
		tmp <- tmp[, c("M norm", "F norm", "inf norm", "AUC", "max F1")]

     	current <- data.frame(Scenario = name[Case], Missing = c(paste0(as.numeric(miss) * 100, "%"), rep(NA, 4)), Estimator = rownames(tmp))
     	current <- cbind(current, tmp)
     	rownames(current) <- NULL
     	all <- rbind(all, current)   
	}
}
all1 <- all[1:30, ]
for(i in name){
	all1[which(all1[,1] == i)[-1], 1] <- NA
} 
print(xtable(all1), include.rownames=FALSE)
# print(xtable(all[25:48, ]), include.rownames=FALSE )
