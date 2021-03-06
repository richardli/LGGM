library(data.table)
library(mnormt)
library(huge)
library(mvtnorm)
library(MCMCpack)
library(truncnorm)
library(tmvtnorm)
library(matrixcalc)
library(ROCR)
library(corrplot)	
library(openVA)
library(VA)
source("functions.r")


for(itr in 1:50){
	if(file.exists(paste0("../data/processed/phmrc", itr, ".rda"))){
		# load(paste0("../data/processed/phmrc", itr, ".rda"))
		next
	}
	dir <- paste0("../experiments/PHMRClong-0-", itr, "-20N/PHMRClong-0-", itr, "-20N_")
	train_sub <- read.csv(paste0("../data/expnew/PS_0_train", itr, ".csv"), header = F)
	delta <- read.csv(paste0("../data/expnew/typePS", itr, "_delta.csv"), header = F)
	test_sub <- read.csv(paste0("../data/expnew/PS_0_test", itr, ".csv"), header = F)
	G <- dim(delta)[1]
	csmf <- rep(1/G, G)
	csmf.train <- csmf
	probbase <- 1 - pnorm(as.matrix(delta))
	P <- dim(probbase)[2]
	causes <- c(test_sub[, 1])
	csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))
	fit.nb <- evalNBprob2(probbase = probbase, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
	
	pnb <- fit.nb$pnb
	pick.nb <- apply(pnb, 1, which.max)

	# Plug in estimator
	test0 <- matrix(NA, dim(test_sub)[1], P)
	test0[test_sub[,-1] == "Y"] <- 1
	test0[test_sub[,-1] == "N"] <- 0
	
	# Latent Gaussian results
	prob <- t(as.matrix(fread(paste0(dir, "assignment_out_mean.txt"))))
	csmf <- as.matrix(fread(paste0(dir, "prob_out.txt")))[, -1]
	sub <- 1:dim(csmf)[2]
	sub <- sub[(length(sub)/2+1):length(sub)]
	sub <- sub[sub%%10 == 0]
	csmf.mean <- apply(csmf[, sub], 1, mean)
	metric <- matrix(NA, 6, 4)
	colnames(metric) <- c("CSMF", "Top 1", "Top 2", "Top 3")
	rownames(metric) <- c("Tariff", "InterVA", "Naive Bayes", "InSilicoVA", "King-Lu", "Gaussian Mixture")
	metric["Naive Bayes", ] <- fit.nb$fitted.nb
	metric["InterVA", ] <- fit.nb$fitted.inter
	metric["Gaussian Mixture", ] <- getAccuracy(prob, test_sub[, 1], csmf.true, csmf.mean)



	## Other methods
	indic.test<- test0
	train0 <- matrix(NA, dim(train_sub)[1], P)
	train0[train_sub[,-1] == "Y"] <- 1
	train0[train_sub[,-1] == "N"] <- 0
	indic.train <- train0
	indic.test[is.na(indic.test)] <- 0
	indic.train[is.na(indic.train)] <- 0
	symcount1 <- apply(indic.test, 2, sum)
	symcount2 <- apply(indic.train, 2, sum)   
	invar <- union(which(symcount1 %in% c(0, dim(indic.test)[1])), which(symcount2 %in% c(0, dim(indic.train)[1])))
	indic.test<- indic.test[, -invar]
	indic.train <- indic.train[, -invar]    
	colnames(indic.test) <- colnames(indic.train) <- paste0("V", 1:dim(indic.train)[2])
	indic.train <- cbind(cod = train_sub[, 1], indic.train)
	indic.test <- cbind(cod = test_sub[, 1], indic.test)
	p <- dim(indic.test)[2]-1
	formula <- as.formula(paste0("cbind(V1+...+V", p, ")~cod"))
	KL <- va(formula, data = list(indic.train, indic.test), nsymp=18) 
	csmf.KL <- KL$est.CSMF
	metric["King-Lu", 1] <- getAccuracy(prob, test_sub[, 1], csmf.true, csmf.KL)[1]

	
	test_sub_1 <- cbind(ID = 1:dim(test_sub)[1], test_sub)
	test_sub_1 <- ConvertData(test_sub_1, yesLabel = "Y", noLabel = c("N"), missLabel=".")
	train_sub_1 <- cbind(ID = 1:dim(train_sub)[1], train_sub)
	train_sub_1 <- ConvertData(train_sub_1, yesLabel = "Y", noLabel = c("N", "."), missLabel = NA)
	colnames(test_sub_1) <- c("ID", "Cause", colnames(probbase))
	colnames(train_sub_1) <- c("ID", "Cause", colnames(probbase))
	fit <- insilico.train2(data = test_sub_1, data.type = "customize", train = train_sub_1, cause = "Cause", causes.table = as.character(1:G), Nsim = 10000, auto.length = FALSE, updateCondProb = FALSE, datacheck=FALSE, burnin = 5000, CondProbNum = probbase, type = "empirical")
	csmf.tmp <- getCSMF(fit)[,1]
	prob.tmp <- getIndivProb(fit)
	metric["InSilicoVA", ] <- getAccuracy(prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)


	fit <- codeVA(data = test_sub_1, data.type = "customize", data.train = train_sub_1, causes.train = "Cause", causes.table = as.character(1:G), model = "Tariff")
	csmf.tmp <- getCSMF(fit)[as.character(1:G)]
	prob.tmp <- getIndivProb(fit)[, as.character(1:G)]
	metric["Tariff", ] <- getAccuracy(-prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)
	print(metric)
	save(metric, file = paste0("../data/processed/phmrc", itr, ".rda"))
	print(itr)
}


library(plyr)
library(ggplot2)
metrics.all <- data.frame(matrix(NA, 50*6*4, 4))
colnames(metrics.all) <- c("Method", "Metric", "Value", "Rep")
counter <- 1
for(itr in 1:50){
	load(paste0("../data/processed/phmrc", itr, ".rda"))
	tmp <- data.frame(Method  = rep(rownames(metric), 4), 
					  Metric  = rep(colnames(metric), each = 6), 
					  Value = as.numeric(metric), 
					  Rep = itr, stringsAsFactors=FALSE)
	metrics.all[counter : (counter + 6*4 - 1), ] <- tmp
	counter <- counter + 6*4 
}

metrics.all$Metric <-factor(metrics.all$Metric, levels= c("CSMF", "Top 1", "Top 2", "Top 3"))
metrics.all$Metric <- revalue(metrics.all$Metric, c("CSMF"="CSMF Accuracy", "Top 1" = "Top Cause Acc", "Top 2" = "Top 2 Cause Acc", "Top 3" = "Top 3 Cause Acc"))
metrics.all$Method <- factor(metrics.all$Method, levels = c("Tariff",  "InterVA" , "Naive Bayes", "InSilicoVA", "King-Lu", "Gaussian Mixture"))

cbPalette <- c('#377eb8','#4daf4a','#984ea3','#ff7f00','#e41a1c', '#a65628')

pdf("../figures/PHMRC_compare_box.pdf", width = 9, height=4)
g <- ggplot(subset(metrics.all, Metric != "Top 2 Cause Acc" & Method != "King-Lu"), aes(x = Method, y = Value, color = Method)) + facet_wrap(~Metric, scales = "free") + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("Accuracy") + scale_color_manual(values=cbPalette)
print(g)
dev.off()
 