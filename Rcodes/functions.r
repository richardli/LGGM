plug_in_estimator <- function(data, type, corr, delta, csmf = NULL, sigma2 = NULL){
	Ntest <- dim(data)[1]
	P <- dim(data)[2]
	G <- dim(delta)[1]
	if(is.null(sigma2)) sigma2 <- rep(0, G)
	lower <- matrix(-Inf, Ntest, P)
	upper <- matrix(Inf, Ntest, P)
	lower[data == 1] <- 0
	upper[data == 0] <- 0
	if(is.null(csmf)) csmf <- rep(1/G, G)
	cont <- which(type == 0)
	bin <- which(type == 1)
	
	pnb <- matrix(NA, Ntest, G)
	corr <- as.matrix(forceSymmetric(.5 * (corr + t(corr))))
	for(i in 1:Ntest){
		for(g in 1:G){
			mean = delta[g, ]
			sigma = corr + diag(sigma2[g], P)
			conti <- cont[which(!is.na(data[i, cont]))]
			if(length(conti) > 0){
				mean <-as.numeric(mean[bin]) + as.numeric(t(solve(sigma[bin, bin])) %*% sigma[bin, conti] %*% (data[i, conti] - mean[conti]) ) 
				sigma <- sigma[bin, bin] - sigma[bin, conti] %*% solve(sigma[conti, conti]) %*% sigma[conti, bin]
				sigma <- (sigma + t(sigma))/2
			}else{
				mean <- as.numeric(mean[bin])
				sigma <- sigma[bin, bin]
				sigma <- (sigma + t(sigma))/2
			}
			pnb[i, g] <- pmvnorm(
						lower=lower[i, bin], 
						upper=upper[i, bin],
						mean=mean, 
						sigma=sigma)[1] * csmf[g]
		}
		if(sum(pnb[i, ]) > 0){
			pnb[i, ] <- pnb[i, ] / sum(pnb[i, ])
		}
		cat(".")
	}
	return(pnb)	
}

naivebayes_eval <- function(type, train, test, train.Y, test.Y, csmf, mean.true){
	P <- length(type)
	G <- dim(mean.true)[1]
	delta <- matrix(0, G, P)
	Ntrain <- dim(train)[1]
	Ntest <- dim(test)[1]
	
	cont <- which(type == 0)
	bin <- which(type == 1)
	probbase <- 1 - pnorm(-mean.true)
	
	# cut off continuous variables by their medians
	if(length(cont) > 0){
		for(j in cont){
			test[, j] <- as.numeric(test[, j] > median(train[, j], na.rm=T))
			train[, j] <- as.numeric(train[, j] > median(train[, j], na.rm=T))
			for(g in 1:G){
				probbase[g, j] <- sum(train[train.Y == g,  j], na.rm=T) / length(!is.na(train[train.Y == g, j]))
				if(is.na(probbase[g, j])) probbase[g, j] = 0.5
			}
		}
	}	

	pnb.ind <- matrix(0, Ntest, G)
	pnb.ind.inter <- matrix(0, Ntest, G)
	for(i in 1:Ntest){
		for(g in 1:G){
			pnb.ind.inter[i, g] <- prod(probbase[g, which(test[i,] ==1)], na.rm = TRUE) #* csmf.hat[g]
			pnb.ind[i, g] <- pnb.ind.inter[i,g] * prod(1 - probbase[g, which(test[i,] == 0)], na.rm = TRUE) #* csmf.hat[g]
		}
		if(sum(pnb.ind[i, ]) > 0) pnb.ind[i, ] <- pnb.ind[i, ] / sum(pnb.ind[i, ])
		if(sum(pnb.ind.inter[i, ]) > 0) pnb.ind.inter[i, ] <- pnb.ind.inter[i, ] / sum(pnb.ind.inter[i, ])
	}
	
	
	out <- rbind(getAccuracy(pnb.ind, test.Y, csmf),
				 getAccuracy(pnb.ind.inter, test.Y, csmf))
	
	rownames(out) <- c("NaiveBayes", "InterVA")
	colnames(out) <- c("CSMF", "top1", "top2", "top3")
	return(out)
}
evalNBprob <- function(probbase, testing, G, csmf = NULL, csmf.true){
		Ntest <- dim(testing)[1]
		pnb.ind <- pnb.ind.inter <- matrix(NA, Ntest, G)
		if(is.null(csmf)){
			csmf.train <- rep(1/G, G)
		}
		for(i in 1:Ntest){
			for(g in 1:G){
				pnb.ind.inter[i, g] <- prod(probbase[g, which(testing[i,-1] == "Y")], na.rm = TRUE)  * csmf.train[g]
				pnb.ind[i, g] <- pnb.ind.inter[i,g] * prod(1 - probbase[g, which(testing[i,-1] == "N")], na.rm = TRUE)  
			}
			if(sum(pnb.ind[i, ]) > 0) pnb.ind[i, ] <- pnb.ind[i, ] / sum(pnb.ind[i, ])
			if(sum(pnb.ind.inter[i, ]) > 0) pnb.ind.inter[i, ] <- pnb.ind.inter[i, ] / sum(pnb.ind.inter[i, ])
			cat(".")
		}

		csmf.nb <- apply(pnb.ind, 2, mean)
		csmf.inter <- apply(pnb.ind.inter, 2, mean)
		fitted.nb <- getAccuracy(pnb.ind, testing[, 1], csmf=csmf.true)[1:4]
		fitted.inter <- getAccuracy(pnb.ind.inter, testing[, 1], csmf=csmf.true)[1:4]
		return(list(csmf.true = csmf.true, csmf.nb = csmf.nb, csmf.inter = csmf.inter, fitted.nb = fitted.nb, fitted.inter = fitted.inter, pnb = pnb.ind, pnb.inter = pnb.ind.inter))
	}


evalNBprob2 <- function(probbase, training, testing, G, csmf = NULL, csmf.true, samepop=TRUE){
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
	

	csmf.nb <- apply(pnb.ind, 2, mean)
	csmf.inter <- apply(pnb.ind.inter, 2, mean)
	csmf.miss <- apply(pnb.ind.miss, 2, mean)
	
	fitted.nb[1] <- csmfacc(csmf.nb, csmf.true)
	fitted.inter[1] <- csmfacc(csmf.inter, csmf.true)
	fitted.miss[1] <- csmfacc(csmf.miss, csmf.true)

	return(list(csmf.true = csmf.true, csmf.nb = csmf.nb, csmf.inter = csmf.inter, fitted.nb = fitted.nb, fitted.inter = fitted.inter, pnb = pnb.ind, pnb.inter = pnb.ind.inter, pnb.miss = pnb.ind.miss, fitted.miss=fitted.miss))
}

csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}


# debug(plug_in_estimators_eval)
plug_in_estimators_eval <- function(corr.true, mean.true, type, corrlist, deltalist, csmflist, train, test, train.Y, test.Y, csmf, oracle=TRUE){
	P <- dim(corr.true)[1]
	G <- dim(mean.true)[1]
	delta <- matrix(0, G, P)
	Ntrain <- dim(train)[1]
	Ntest <- dim(test)[1]
	for(g in 1:G){
		phat <-  apply(train[train.Y == g,  , drop=F], 2, function(x){sum(x==1, na.rm = TRUE)}) /  apply(train[train.Y == g, , drop=F], 2, function(x){sum(!is.na(x))})
		delta[g,] <- -qnorm(1 - phat)			
	}
	csmf.hat <- table(c(1:length(csmf), train.Y)) - 1
	csmf.hat <- csmf.hat/sum(csmf.hat)
	if(length(corrlist) > 1){
		if(is.null(csmflist)){
			csmflist <- vector("list", length(corrlist))
		}
		for(i in 1:length(csmflist)){
			if(is.null(csmflist[[i]])) csmflist[[i]] <- csmf.hat
		}		
	}
	delta[delta == -Inf] <- -1e6
	delta[delta == Inf] <- 1e6
	
	cont <- which(type == 0)
	bin <- which(type == 1)
	lower <- matrix(-Inf, Ntest, P)
	upper <- matrix(Inf, Ntest, P)
	lower[test == 1] <- 0
	upper[test == 0] <- 0
	Nplugin <- 0
	names <- NULL
	pnblist <- NULL

	if(!is.null(corrlist)){
		Nplugin <- length(corrlist)
		pnblist <- vector("list", Nplugin+1)
		for(k in 1:Nplugin){
			corr <- corrlist[[k]]
			pnb <- matrix(NA, Ntest, G)
			thisdelta <- deltalist[[k]]
			if(is.null(thisdelta)){thisdelta <- delta}
			if(class(corr) == "matrix"){
				corr <- as.matrix(forceSymmetric(.5 * (corr + t(corr))))
				for(i in 1:Ntest){
					for(g in 1:G){
						mean = thisdelta[g, ]
						sigma = corr
						conti <- cont[which(!is.na(test[i, cont]))]
						if(length(conti) > 0){
							mean <-as.numeric(mean[bin]) + as.numeric(t(solve(sigma[bin, bin])) %*% sigma[bin, conti] %*% (test[i, conti] - mean[conti]) ) 
							sigma <- sigma[bin, bin] - sigma[bin, conti] %*% solve(sigma[conti, conti]) %*% sigma[conti, bin]
							sigma <- (sigma + t(sigma))/2
						}else{
							mean <- as.numeric(mean[bin])
							sigma <- sigma[bin, bin]
							sigma <- (sigma + t(sigma))/2
						}
						pnb[i, g] <- pmvnorm(
									lower=lower[i, bin], 
									upper=upper[i, bin],
									mean=mean, 
									sigma=sigma)[1] #* csmflist[[k]][g]
					}
					if(sum(pnb[i, ]) > 0){
						pnb[i, ] <- pnb[i, ] / sum(pnb[i, ])
					}
					cat(".")
				}	
			}else{
				for(i in 1:length(corr)){
					corr[[i]] <- as.matrix(forceSymmetric(.5 * (corr[[i]]+ t(corr[[i]]))))
				}
				for(i in 1:Ntest){
					for(g in 1:G){
						mean = thisdelta[g, ]
						sigma = corr[[g]]
						conti <- cont[which(!is.na(test[i, cont]))]
						if(length(conti) > 0){
							mean <- as.numeric(mean[bin]) + as.numeric(t(solve(sigma[bin, bin])) %*% sigma[bin, conti] %*% (test[i, conti] - mean[conti]) ) 
							sigma <- sigma[bin, bin] - sigma[bin, conti] %*% solve(sigma[conti, conti]) %*% sigma[conti, bin]
							sigma <- (sigma + t(sigma))/2
						}else{
							mean <- as.numeric(mean[bin])
							sigma <- sigma[bin, bin]
							sigma <- (sigma + t(sigma))/2
						}
						pnb[i, g] <- pmvnorm(
									lower=lower[i, bin], 
									upper=upper[i, bin],
									mean=mean, 
									sigma=sigma)[1] #* csmflist[[k]][g]
					}
					if(sum(pnb[i, ]) > 0){
						pnb[i, ] <- pnb[i, ] / sum(pnb[i, ])
					}
					cat(".")
				}
			}
			pnblist[[k]] <- pnb
			print("Finish one")
		}
		if(is.null(names(corrlist))){
			names <- 1:Nplugin
		}else{
			names <- names(corrlist)
		}
	}
		
	if(oracle){
		# oracle case:
		pnb <- matrix(NA, Ntest, G)
		corr.true <- as.matrix(forceSymmetric(.5 * (corr.true+ t(corr.true))))
		for(i in 1:Ntest){
			for(g in 1:G){
				mean = mean.true[g, ]
				sigma = corr.true
				conti <- cont[which(!is.na(test[i, cont]))]
				if(length(conti) > 0){
					mean <- as.numeric(mean[bin]) + as.numeric(t(solve(sigma[bin, bin])) %*% sigma[bin, conti] %*% (test[i, conti] - mean[conti]) ) 
					sigma <- sigma[bin, bin] - sigma[bin, conti] %*% solve(sigma[conti, conti]) %*% sigma[conti, bin]
					sigma <- (sigma + t(sigma))/2
				}else{
					mean <- as.numeric(mean[bin])
					sigma <- sigma[bin, bin]
					sigma <- (sigma + t(sigma))/2
				}
				pnb[i, g] <- pmvnorm(
							lower=lower[i, bin], 
							upper=upper[i, bin],
							mean=mean, 
							sigma=sigma)[1] * as.numeric(csmf)[g]
			}
			if(sum(pnb[i, ]) > 0){
				pnb[i, ] <- pnb[i, ] / sum(pnb[i, ])
			}
			cat(".")
		}
		print("Finish Oracle")
		pnblist[[Nplugin+1]] <- pnb
	}

		

	# probbase <- matrix(0, G, P)
	# for(g in 1:G){
	# 	probbase[g, ] <- apply(train[train.Y == g,  ], 2, function(x){sum(x==1, na.rm = TRUE)}) /  apply(train[train.Y == g, ], 2, function(x){sum(!is.na(x))})
	# }
	probbase <- 1 - pnorm(-mean.true)
	
	# cut off continuous variables by their medians
	if(length(cont) > 0){
		for(j in cont){
			test[, j] <- as.numeric(test[, j] > median(train[, j], na.rm=T))
			train[, j] <- as.numeric(train[, j] > median(train[, j], na.rm=T))
			for(g in 1:G){
				probbase[g, j] <- sum(train[train.Y == g,  j], na.rm=T) / length(!is.na(train[train.Y == g, j]))
				if(is.na(probbase[g, j])) probbase[g, j] = 0.5
			}
		}
	}	

	pnb.ind <- matrix(0, Ntest, G)
	pnb.ind.inter <- matrix(0, Ntest, G)
	for(i in 1:Ntest){
		for(g in 1:G){
			pnb.ind.inter[i, g] <- prod(probbase[g, which(test[i,] ==1)], na.rm = TRUE) #* csmf.hat[g]
			pnb.ind[i, g] <- pnb.ind.inter[i,g] * prod(1 - probbase[g, which(test[i,] == 0)], na.rm = TRUE) #* csmf.hat[g]
		}
		if(sum(pnb.ind[i, ]) > 0) pnb.ind[i, ] <- pnb.ind[i, ] / sum(pnb.ind[i, ])
		if(sum(pnb.ind.inter[i, ]) > 0) pnb.ind.inter[i, ] <- pnb.ind.inter[i, ] / sum(pnb.ind.inter[i, ])
	}
	
	out <- NULL
	if(length(pnblist) > 0){
		for(k in 1:length(pnblist)){
			out <- rbind(out, getAccuracy(pnblist[[k]], test.Y, csmf))
		}
	}
	out <- rbind(out,
				 getAccuracy(pnb.ind, test.Y, csmf),
				 getAccuracy(pnb.ind.inter, test.Y, csmf))
	
	if(oracle){
		rownames(out) <- c(names, "Oracle", "NaiveBayes", "InterVA")
	}else{
		rownames(out) <- c(names, "NaiveBayes", "InterVA")
	}
	colnames(out) <- c("CSMF", "top1", "top2", "top3", 
					   "CSMF-p", "top1-p", "top2-p", "top3-p")
	return(list(metrics = out, pnb = pnblist, pnb.nb = pnb.ind, pnb.inter = pnb.ind.inter, csmf = csmf, train.Y = train.Y, test.Y = test.Y, train.X = train, test.X = test, corr.true = corr.true, mean.true = mean.true))
}



getAccuracy <- function(pnb, membership, csmf, csmf.fit = NULL){
	n <- dim(pnb)[1]
	csmf <- as.vector(csmf)
	
	if(is.null(csmf.fit)) csmf.fit <- apply(pnb, 2, mean)
	csmf.acc <- 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))
	pick1 <- sum(apply(pnb, 1, which.max) == membership)

	which.ordern <- function(x, n, isMax){
			tmp <- order(x, decreasing = isMax)[n]
			if(is.na(x[tmp])){return(NA)}
			return(tmp)
		}

	pick2<- sum(apply(pnb, 1, which.ordern, 2, TRUE)==membership, na.rm = TRUE)
	pick3<- sum(apply(pnb, 1, which.ordern, 3, TRUE)==membership, na.rm = TRUE)
	flat <- c(csmf.acc, pick1/n, (pick1+pick2)/n, (pick1+pick2+pick3)/n)
	
	return(flat)
}



# calculate HBIC for a path of precision matrices
HBIC <- function(R, icov, n, d){
	out <- rep(0, length(icov))
	for(i in 1:length(icov)){
		s0 <- icov[[i]]
		diag(s0) <- 0
		nEdge <- sum(s0 != 0) / 2
		out[i] <- sum(diag(R %*% icov[[i]])) - log(det(icov[[i]]))
		out[i] <- out[i] + log(log(n)) * log(d) / n * nEdge
		# this is regular BIC
		# out[i] <- out[i] + log(n) / n * nEdge
	}
	return(out)
}

# calculate HBIC for a path of selection
HBIC_thre <- function(icov, thre, n, d){
	R <- cov2cor(solve(icov))
	K <- length(unique(as.vector(thre)))
    cutoff <- sort(unique(as.vector(thre)), decreasing = TRUE)
   
	out <- rep(0, K)
	for(i in 1:K){
		s0 <- matrix(1, dim(thre)[1], dim(thre)[2])
		s0[thre < cutoff[i]] <- 0
		diag(s0) <- 0
		nEdge <- sum(s0 != 0) / 2
		diag(s0) <- 1
		tmp <- icov * s0
		out[i] <- sum(diag(R %*% tmp)) - log(det(tmp))
		out[i] <- out[i] + log(log(n)) * log(d) / n * nEdge
		# this is regular BIC
		# out[i] <- out[i] + log(n) / n * nEdge
	}
	return(list(out, cutoff))
}

getROC <- function(outlist, fprlist, s0){
	K <- length(outlist)
	M <- length(fprlist)

	out <- matrix(0, M, K)		
	
	s00 <- s0 * 2 - 1
	s1 <- s00 * lower.tri(s00)
	n.one <- length(which(s1 == 1))
	n.zero <- length(which(s1 == -1))
	for(j in 1:K){
		out0 <- outlist[[j]]
		tpr <- fpr <- rep(0, length(out0))
		for(i in 1:length(out0)){
			est <- lower.tri(out0[[i]]) * out0[[i]]
			est <- est * 2 - 1
			tpr[i] <- length(intersect(which(est == 1), which(s1 == 1))) / n.one
			fpr[i] <- length(intersect(which(est == 1), which(s1 == -1))) / n.zero
		}
		fpr <- c(fpr, 1)
		tpr <- c(tpr, 1)
		out[, j] <- approx(fpr, tpr, xout = fprlist)$y
		if(max(out[, j], na.rm = TRUE) == 1){
			reach <- which(out[, j] == 1)[1]
			out[reach:M, j] <- 1
		}		
	}

	out <- cbind(fprlist, out)
	fitnames <- names(outlist)
	if(is.null(fitnames)){fitnames <- "TPR"}
	colnames(out) <- c("FPR", fitnames)
	out <- data.frame(out)
	return(out)
}

get4norm <- function(list, cov, normalize=TRUE){
	mn <- NULL
	sn <- NULL
	fn <- NULL
	infn <- NULL
	for(i in 1:length(list)){
		if(!normalize){
			m1 <- as.matrix(list[[i]])
			m2 <- cov	
		}else{
			m1 <- cov2cor(as.matrix(list[[i]]))
			m2 <- cov2cor(cov)
		}
		mtemp <- norm( m1 - m2, type = "m")
		stemp <- base::norm(m1 - m2, type = "2")
		ftemp <- norm(m1 - m2, type = "f")
		inftemp <- max(apply(m1 - m2, 1, function(x){sum(abs(x))}))
		mn <- c(mn, mtemp)
		sn <- c(sn, stemp)
		fn <- c(fn, ftemp)
		infn <- c(infn, inftemp)
	}
	out <- cbind(mn, sn, fn, infn)
	colnames(out) <- c("M norm", "S norm", "F norm", "inf norm")
	return(out)
}

# prob is a probability matrix to threshold,
# function adapted from huge.roc
rocf1 <- function(prob, theta, verbose = FALSE){
	gcinfo(verbose = FALSE)
    ROC = list()
    theta = as.matrix(theta)
    d = ncol(theta)
    pos.total = sum(theta != 0)
    neg.total = d * (d - 1) - pos.total
    if (verbose) 
        cat("Adapted: Computing F1 scores, false positive rates and true positive rates....")
    K <- length(unique(as.vector(prob)))
    cutoff <- sort(unique(as.vector(prob)), decreasing = TRUE)
    ROC$tp = rep(0, K)
    ROC$fp = rep(0, K)
    ROC$F1 = rep(0, K)
    for (r in 1:K) {
        tmp = as.matrix(prob > cutoff[r])
        tp.all = (theta != 0) * (tmp != 0)
        diag(tp.all) = 0
        ROC$tp[r] <- sum(tp.all != 0)/pos.total
        fp.all = (theta == 0) * (tmp != 0)
        diag(fp.all) = 0
        ROC$fp[r] <- sum(fp.all != 0)/neg.total
        fn = 1 - ROC$tp[r]
        precision = ROC$tp[r]/(ROC$tp[r] + ROC$fp[r])
        recall = ROC$tp[r]/(ROC$tp[r] + fn)
        ROC$F1[r] = 2 * precision * recall/(precision + recall)
        if (is.na(ROC$F1[r])) 
            ROC$F1[r] = 0
    }
    if (verbose) 
        cat("done.\n")
    rm(precision, recall, tp.all, fp.all, prob, theta, fn)
    gc()
    ord.fp = order(ROC$fp)
    tmp1 = ROC$fp[ord.fp]
    tmp2 = ROC$tp[ord.fp]
    # par(mfrow = c(1, 1))
    # plot(tmp1, tmp2, type = "b", main = "ROC Curve", xlab = "False Postive Rate", 
    #     ylab = "True Postive Rate", ylim = c(0, 1))
    ROC$AUC = sum(diff(tmp1) * (tmp2[-1] + tmp2[-length(tmp2)]))/2
    rm(ord.fp, tmp1, tmp2)
    gc()
    class(ROC) = "roc"
    return(ROC)

}

## Missing data is represented by NA
getFanEstimator <- function(X, delta){
	Phi <- function(a, b, rho){
		pmnorm(mean = c(0, 0), varcov = matrix(c(1, rho, rho, 1), 2, 2), x = c(a, b))	
	}
	abserror <- function(rho, a, b, int){
		abs(int - Phi(a, b, rho))
	}
	Phiinv <- function(a, b, int){
		tmp <- optimize(f = abserror, interval = c(-1, 1), a = a, b = b, int = int)
		return(list(rho = tmp$minimum, error = tmp$objective))
	}

	X <- as.matrix(X)
	N <- dim(X)[1]
	M <- dim(X)[2]

	tauhat <- matrix(0, M, M)
	for(j in 1:M){
		for(k in 1:M){
			na <- sum(X[, j] * X[, k], na.rm = TRUE)
			nd <- sum((1-X[, j]) * (1-X[, k]), na.rm = TRUE)
			nb <- sum((1-X[, j]) * X[, k], na.rm = TRUE)
			nc <- sum(X[, j] * (1-X[, k]), na.rm = TRUE)
			samplesize <- (na + nb + nc + nd)
			if(samplesize <= 1){
				tauhat[j, k] <- 0
			}else{
				tauhat[j, k] <- 2 * (na * nd - nb * nc) /samplesize / (samplesize- 1)
			}
		}
	}
	if(is.null(delta)){
		phat <-  apply(X, 2, function(x){sum(x==1, na.rm = TRUE)}) /  apply(X, 2, function(x){sum(!is.na(x))})
		delta <- qnorm(1 - phat)
	}else{
		phat <- 1 - pnorm(delta)		
	}

	Rhat <- matrix(0, M, M)
	diag(Rhat) <- 1

	for(j in 1:M){
		for(k in 1:M){
			if(j == k) next
			tmp <- Phiinv(delta[j], delta[k], tauhat[j, k]/2 + (1-phat[j]) * (1-phat[k]))
			Rhat[j, k] <- tmp$rho
		}
	}

	return(Rhat)

}

 # type = 0 -> continuous
 # type = 1 -> binary
getFanEstimatorMix <- function(X, delta, type){
	Phi <- function(a, b, rho){
		pmnorm(mean = c(0, 0), varcov = matrix(c(1, rho, rho, 1), 2, 2), x = c(a, b))	
	}
	abserror <- function(rho, a, b, int){
		abs(int - Phi(a, b, rho))
	}
	Phiinv <- function(a, b, int){
		tmp <- optimize(f = abserror, interval = c(-1, 1), a = a, b = b, int = int)
		return(list(rho = tmp$minimum, error = tmp$objective))
	}

	N <- dim(X)[1]
	M <- dim(X)[2]

	tauhat <- matrix(0, M, M)
	for(j in 1:M){
		for(k in 1:M){
			if(type[j] == 0 || type[k] == 0){
				samplesize <- sum((!is.na(X[,j])) * (!is.na(X[,k])))
				if(samplesize <= 1){
					tauhat[j,k]=0
				}else{
					# sgnsum <- 0
					# for(i in 1 : (N-1)){
					# 	sgnsum <- sgnsum + sum( sign((X[i,j] - X[(i+1):N, j]) * (X[i,k] - X[(i+1):N, k])), na.rm = TRUE)
					# }
					# tauhat[j, k] <- 2/(samplesize * (samplesize - 1)) * sgnsum
					tauhat[j,k] <- cor(X[, c(j,k)], method="kendall", use="complete.obs")[1,2]
				}

			}else{
				na <- sum(X[, j] * X[, k], na.rm = TRUE)
				nd <- sum((1-X[, j]) * (1-X[, k]), na.rm = TRUE)
				nb <- sum((1-X[, j]) * X[, k], na.rm = TRUE)
				nc <- sum(X[, j] * (1-X[, k]), na.rm = TRUE)
				samplesize <- (na + nb + nc + nd)
				if(samplesize <= 1){
					tauhat[j, k] <- 0
				}else{
					tauhat[j, k] <- 2 * (na * nd - nb * nc) /samplesize / (samplesize- 1)
				}
			}
		}
	}
	
	if(is.null(delta)){
		phat <-  apply(X, 2, function(x){sum(x==1, na.rm = TRUE)}) /  apply(X, 2, function(x){sum(!is.na(x))})
		phat[is.na(phat)] <- 0.5
		delta <- qnorm(1 - phat)
	}else{
		phat <- 1 - pnorm(delta)
	}

	Rhat <- matrix(0, M, M)
	diag(Rhat) <- 1

	for(j in 1:M){
		for(k in 1:M){
			if(j == k) next
			if(type[j] == 0 && type[k] == 0){
				Rhat[j, k] <- sin(pi * tauhat[j,k] / 2)
			
			}else if(type[j] == 0 && type[k] != 0){
				tmp <- Phiinv(delta[k], 0, 
						tauhat[j, k]/4 + (1-phat[k])/2)
				Rhat[j, k] <- tmp$rho * sqrt(2)	

			}else if(type[j] != 0 && type[k] == 0){
				tmp <- Phiinv(delta[j], 0, 
						tauhat[j, k]/4 + (1-phat[j])/2)
				Rhat[j, k] <- tmp$rho * sqrt(2)	
			# both binary
			}else{
				tmp <- Phiinv(delta[j], delta[k], tauhat[j, k]/2 + (1-phat[j]) * (1-phat[k]))
				Rhat[j, k] <- tmp$rho					
			}
		}
	}

	return(Rhat)

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



insilico.train2<- function(data, train, cause, causes.table = NULL, thre = 0.95, type = c("quantile", "fixed", "empirical")[1], isNumeric = FALSE, updateCondProb = TRUE, keepProbbase.level = TRUE,  CondProb = NULL, CondProbNum = NULL, datacheck = TRUE, datacheck.missing = TRUE, warning.write = FALSE, external.sep = TRUE, Nsim = 4000, thin = 10, burnin = 2000, auto.length = TRUE, conv.csmf = 0.02, jump.scale = 0.1, levels.prior = NULL, levels.strength = NULL, trunc.min = 0.0001, trunc.max = 0.9999, subpop = NULL, java_option = "-Xmx1g", seed = 1, phy.code = NULL, phy.cat = NULL, phy.unknown = NULL, phy.external = NULL, phy.debias = NULL, exclude.impossible.cause = TRUE, impossible.combination = NULL, indiv.CI = NULL, ...){ 
	  
	  # handling changes throughout time
	  args <- as.list(match.call())
	  if(!is.null(args$length.sim)){
	  	Nsim <- args$length.sim
	  	message("length.sim argument is replaced with Nsim argument, will remove in later versions.\n")
	  }



	if(type == "empirical"){
		cat("Empirical conditional probabilities are used, so updateCondProb is forced to be FALSE.")
		updateCondProb <- FALSE
	}

	if(is.null(CondProbNum)){
		prob.learn <- extract.prob(train = train, 
						  gs = cause, 
						  gstable = causes.table, 
						  thre = thre, 
						  type = type, 
						  isNumeric = isNumeric)
		# remove unused symptoms
		col.exist <- c(colnames(data)[1], cause, colnames(prob.learn$symps.train))
		remove <- which(colnames(data) %in% col.exist == FALSE)
		if(length(remove) > 0){
			warning(paste(length(remove), "symptoms deleted from testing data to match training data:", 
				paste(colnames(data)[remove], collapse = ", ")),
				immediate. = TRUE)	
			data <- data[, -remove]
		}

	}
	if(is.null(CondProbNum)){
		if(updateCondProb){
			probbase.touse <- prob.learn$cond.prob.alpha
			CondProbNum <- NULL
		}else{
			probbase.touse <- prob.learn$cond.prob
			CondProbNum <- prob.learn$cond.prob
		}
	}else{
		probbase.touse <- t(CondProbNum)
		prob.learn <- NULL
		prob.learn$cond.prob <- matrix(NA, 1, length(causes.table))
		colnames(prob.learn$cond.prob) <- causes.table
	}



	# default levels.strength for two different P(S|C) extraction
	if(is.null(levels.strength)){
	  if(type == "empirical"){
	    levels.strength <- 1 # doesn't matter anyway
	  }else if(type == "fixed"){
	    levels.strength <- 1
	  }else if(type == "quantile"){
	    levels.strength <- 0.01
	  }
	}
	# Notice that by default, data.type is set to WHO 2012. 
	# This is only consequential to the codings are done. 
	# It does not take over the probbase from training data, 
	#   because customization.dev is set to TRUE.
	fit <- insilico.fit(data = data, 
						isNumeric = isNumeric, 
						updateCondProb = updateCondProb, 
						keepProbbase.level = keepProbbase.level, 
						CondProb = CondProb, 
						CondProbNum = CondProbNum, 
						datacheck = FALSE, 
						datacheck.missing = FALSE, 
						warning.write = FALSE, 
						external.sep = FALSE, 
						Nsim = Nsim, 
						thin = thin, 
						burnin = burnin, 
						auto.length = auto.length, 
						conv.csmf = conv.csmf, 
						jump.scale = jump.scale, 
						levels.prior = levels.prior, 
						levels.strength = levels.strength, 
						trunc.min = trunc.min, 
						trunc.max = trunc.max, 
						subpop = subpop, 
						java_option = java_option, 
						seed = seed, 
						phy.code = phy.code, 
						phy.cat = phy.cat, 
						phy.unknown = phy.unknown, 
						phy.external = phy.external, 
						phy.debias = phy.debias, 
						exclude.impossible.cause = FALSE, 
						indiv.CI = indiv.CI, 
						impossible.combination = impossible.combination,
						
						customization.dev = TRUE, 
						Probbase_by_symp.dev = FALSE, 
						probbase.dev = probbase.touse, 
						table.dev = prob.learn$table.alpha, 
						table.num.dev = prob.learn$table.num, 
						gstable.dev = colnames(prob.learn$cond.prob), 
						nlevel.dev = 15
						)
	return(fit)  	
} 