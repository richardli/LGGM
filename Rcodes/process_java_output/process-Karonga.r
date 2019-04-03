##
## Process Karonga data fit example
##	Working directory set to Rcodes/
##	setwd("../")
##

## --------------------------------------------------------------------##
## Task 1: Prediction
## --------------------------------------------------------------------##
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
library(xtable)
source("functions.r")

case <- c("4RR")
dir <- paste0("../experiments/Karonga2019-", case, "/Karonga2019-", case, "_")
tmp <- as.matrix(fread(paste0(dir, "corr_out.txt")))
Nitr <- dim(tmp)[1]/dim(tmp)[2]
P <- dim(tmp)[2]
corr.mean <- array(0, dim=c(P, P))
counter <- 1
for(i in 1:Nitr){
	corr.mean <- corr.mean + as.matrix(tmp[counter:(counter + P-1), ])/Nitr
	counter <- counter + P
}
tmp <- as.matrix(fread(paste0(dir, "invcorr_out.txt")))
prec.mean <- array(0, dim=c(P, P))
counter <- 1
for(i in 1:Nitr){
	prec.mean <- prec.mean + as.matrix(tmp[counter:(counter + P-1), ])/Nitr
	counter <- counter + P
}
G <- 16
mean <- as.matrix(fread(paste0(dir, "mean_out.txt")))
Nitr <- dim(mean)[1]/G
P <- dim(tmp)[2]
delta <- matrix(0, P, G)
counter <- 1
for(i in 1:Nitr){
	delta <- delta + t(as.matrix(tmp[counter:(counter + G-1), ]))/Nitr
	counter <- counter + P
}
csmf <- as.matrix(fread(paste0(dir, "prob_out.txt")))[, -1]
sub <- 1:dim(csmf)[2]
sub <- sub[(length(sub)/2+1):length(sub)]
sub <- sub[sub%%10 == 0]
csmf.mean <- apply(csmf[, sub], 1, mean)

# Get results
prob <- t(as.matrix(fread(paste0(dir, "assignment_out_mean.txt"))))
inclusion <- as.matrix(fread(paste0(dir, "inclusion_out.txt"), sep = ","))

assign <- apply(prob, 1, which.max)
assign2 <- apply(prob, 1, function(x){order(x, decreasing=TRUE)[2]})
assign3 <- apply(prob, 1, function(x){order(x, decreasing=TRUE)[3]})

# Get raw data
delta <- read.csv(paste0("../data/expnew/typeK3_delta.csv"), header = F)
train_sub <- read.csv(paste0("../data/expnew/K_train0.csv"), header = F)
test_sub <- read.csv(paste0("../data/expnew/K_test0.csv"), header = F)
probbase <- 1 - pnorm(as.matrix(delta))
G <- dim(probbase)[1]
P <- dim(probbase)[2]
csmf.train <- rep(1/G, G)
causes <- test_sub[, 1]
csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))
fit.nb <- evalNBprob2(probbase = probbase, training = train_sub, testing = test_sub, G, csmf = NULL, csmf.true = csmf.true, samepop = FALSE)
pnb <- fit.nb$pnb
pick.nb <- apply(pnb, 1, which.max)

# Plug in estimator
test0 <- matrix(NA, dim(test_sub)[1], P)
test0[test_sub[,-1] == "Y"] <- 1
test0[test_sub[,-1] == "N"] <- 0


metric <- matrix(NA, 6, 4)
colnames(metric) <- c("CSMF", "Top 1", "Top 2", "Top 3")
rownames(metric) <- c("Tariff", "InterVA", "Naive Bayes", "InSilicoVA", "King-Lu", "Gaussian Mixture")
metric["Naive Bayes", ] <- fit.nb$fitted.nb
metric["InterVA", ] <- fit.nb$fitted.inter
metric["Gaussian Mixture", ] <- getAccuracy(prob, test_sub[, 1], csmf.true, csmf.mean)

###########################################################################
# install.packages("VA_0.9-2.12.tar.gz", repos = NULL, type = "source")
library(VA)
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

## Other methods
library(openVA)
test_sub_1 <- cbind(ID = 1:dim(test_sub)[1], test_sub)
test_sub_1 <- ConvertData(test_sub_1, yesLabel = "Y", noLabel = c("N"), missLabel =  ".")
train_sub_1 <- cbind(ID = 1:dim(train_sub)[1], train_sub)
train_sub_1 <- ConvertData(train_sub_1, yesLabel = "Y", noLabel = c("N", "."), missLabel = NA)
colnames(test_sub_1) <- c("ID", "Cause", colnames(probbase))
colnames(train_sub_1) <- c("ID", "Cause", colnames(probbase))
fit1 <- insilico.train2(data = test_sub_1, data.type = "customize", train = train_sub_1, cause = "Cause", causes.table = as.character(1:G), Nsim = 10000, auto.length = FALSE, updateCondProb = FALSE, datacheck=FALSE, burnin = 5000, CondProbNum = probbase, type = "empirical")
csmf.tmp <- getCSMF(fit1)[,1]
prob.tmp <- getIndivProb(fit1)
metric["InSilicoVA", ] <- getAccuracy(prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)


fit2 <- codeVA(data = test_sub_1, data.type = "customize", data.train = train_sub_1, causes.train = "Cause", causes.table = as.character(1:G), model = "Tariff")
csmf.tmp <- getCSMF(fit2)[as.character(1:G)]
prob.tmp <- getIndivProb(fit2)[, as.character(1:G)]
metric["Tariff", ] <- getAccuracy(-prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)

csmf.plot2 <- data.frame(Method = "Tariff", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = as.numeric(getCSMF(fit2)[as.character(1:G)]), 
						   lower = NA, upper = NA
						   )
csmf.plot2 <- rbind(csmf.plot2, data.frame(Method = "InterVA", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = fit.nb$csmf.inter, 
						   lower = NA, upper = NA
						   ))
csmf.plot2 <- rbind(csmf.plot2, data.frame(Method = "Naive Bayes", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = fit.nb$csmf.nb, 
						   lower = NA, upper = NA
						   ))
csmf.plot2 <- rbind(csmf.plot2, data.frame(Method = "InSilicoVA", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = getCSMF(fit1)[,1], 
						   lower = getCSMF(fit1)[,3], upper = getCSMF(fit1)[,5]
						   ))
csmf.plot2 <- rbind(csmf.plot2, data.frame(Method = "King-Lu", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = KL$est.CSMF, 
						   lower = NA, upper = NA
						   ))
csmf.plot2 <- rbind(csmf.plot2, data.frame(Method = "Gaussian Mixture", 
						   cause = 1:G,
						   truth = csmf.true,
						   value = apply(csmf[, sub], 1, mean), 
						   lower = apply(csmf[, sub], 1, quantile, 0.025), 
						   upper = apply(csmf[, sub], 1, quantile, 0.975)
						   ))
causes <- c("TB/AIDS",                                       
			"Cardiovascular disorder",                       
			"Acute febrile illness",                        
			"Genito urinary disorders",                     
			"Gastro intestinal disorder",                   
			"Central nervous system disorder",              
			"Endocrine disorders",                          
			"Neoplasm",                                     
			"Anaemia",                                      
			"NCD - unspecifiable or other unlisted",        
			"Maternal",                                     
			"Nutritional disorder",                         
			"Communicable - unspecifiable or other unlisted",
			"Diarrhoeal disease without fever",             
			"Respiratory disorder", 
			"Cause of death unknown")  
csmf.plot2$cause <- causes[csmf.plot2$cause]
library(ggrepel)
csmf.plot2 <- subset(csmf.plot2, Method != "King-Lu")
pdf("../Figures/Karonga_csmf.pdf", width = 10, height=10/3*2)
lim.csmf <- range(csmf.plot2[, -c(1,2)], na.rm=TRUE)
g <- ggplot(data = subset(csmf.plot2, Method != ""))
g <- g + geom_errorbar(aes(x=truth, y=value, ymin=lower, ymax=upper), alpha = 0.8, color = "#377eb8")
g <- g + geom_point(aes(x=truth, y=value), col="#d7301f", size = 1.8)
# g <- g + geom_text(aes(x=truth, y=value, label=cause), col="red", size = 2.5, nudge_y = 0.00, nudge_x = 0.01)
g <- g + geom_text_repel(data = subset(csmf.plot2, truth > 0.05), aes(truth, value, label=cause), color = "#d73027", size = 2.5, segment.alpha=0, force = 1.2)
g <- g + facet_wrap(~Method)
g <- g + geom_abline(slope=1,intercept=0, linetype="longdash")
g <- g + xlim(lim.csmf) + ylim(lim.csmf) + theme_bw()
g <- g + xlab("True CSMF") + ylab("Estimated CSMF")
g
dev.off()

###########################################################################
print(metric)
xtable(metric, digits=3)
source("process-Karonga-plots.r")


## --------------------------------------------------------------------##
## Task 2: Convergence with multiple chains
## --------------------------------------------------------------------##

library(rstan)
cases <- c("1RR","2RR", "3RR", "4RR")
for(ii in 1:length(cases)){
	case <- cases[ii]
	dir <- paste0("../experiments/Karonga2019-", case, "/Karonga2019-", case, "_")
	csmf <- as.matrix(fread(paste0(dir, "prob_out.txt")))[, -1]
	size <- as.numeric(fread(paste0(dir, "inclusion_iteration_out.txt")))[-1]
	csmf <- csmf[, (1:dim(csmf)[2])%%10==0 ]
	Nitr <- dim(csmf)[2]
	if(ii == 1){
		csmfall <- array(0, dim = c(length(cases), dim(csmf)))
		sizeall <- matrix(0, length(cases), length(size))
	}
	# order <- order(apply(csmf, 1, mean), decreasing = TRUE)
	# csmf2 <- csmf[, (round(dim(csmf)[2]/2):dim(csmf)[2]) ]
	csmfall[ii, , ] <- csmf
	sizeall[ii, ] <- size
}
G <- dim(csmf)[1]
toplot <- (1:Nitr)[-c(1:(Nitr/2))]
order <- order(apply(csmfall[,,toplot], 2, mean), decreasing = TRUE)
dimnames(csmfall)[[2]] <- causes
#iterations * chains * parameters
tt <- aperm(csmfall[,order,], c(3, 1, 2))
tab <- rstan::monitor(tt, digits_summary = 3)
tab <- as.matrix(tab)
tab <- rbind(as.matrix(rstan::monitor(t(sizeall) , digits_summary = 3)), tab)
rownames(tab)[1] <- "Graph size"
library(xtable)
xtable(tab[, c(1, 10), drop=FALSE])


cols <- c( "red", "blue", "green","black")
toplot <- (1:dim(sizeall)[2])
# toplot <- toplot[-c(1:(length(toplot)/2))]
pdf("../figures/Karonga-size-multichain.pdf", width = 8, height = 4)
par(mfrow = c(1, 1))
for(ii in 1:dim(sizeall)[1]){
	lim <- range(sizeall)
	if(ii==1) plot(toplot, sizeall[ii, toplot], type = "l", main = "Graph size", xlab = "Iteration", ylab = "Size", col=alpha(cols[ii], 0.6), ylim = range(sizeall[, toplot]))
	if(ii > 1) lines(toplot, sizeall[ii, toplot], type = "l", col=alpha(cols[ii], 0.6))
}

pdf("../figures/Karonga-csmf-multichain.pdf", width = 8*2, height = 8)
toplot <- (1:Nitr)#[-c(1:(Nitr/2))]
par(mfrow = c(4, 4))
for(i in 1:dim(csmfall)[2]){
	for(ii in 1:length(cases)){
		lim <- range(c(csmfall[, order, toplot]))
		if(ii==1) plot(toplot, csmfall[ii, order[i], toplot], type = "l", main = causes[order[i]], ylim = lim, xlab = "Iteration", ylab = "CSMF", col=alpha(cols[ii], 0.6))
		if(ii > 1) lines(toplot, csmfall[ii, order[i], toplot], type = "l", col=alpha(cols[ii], 0.6))
	}	
}
dev.off()

pdf("../figures/Karonga-csmf-multichain2.pdf", width = 8*2, height = 8)
toplot <- (1:Nitr)[-c(1:(Nitr/2))]
par(mfrow = c(4, 4))
for(i in 1:dim(csmfall)[2]){
	for(ii in 1:length(cases)){
		# lim <- range(csmfall)
		lim <- range(c(csmfall[, order[i], toplot]))
		if(ii==1) plot(toplot, csmfall[ii, order[i], toplot], type = "l", main = causes[order[i]], ylim = lim, xlab = "Iteration", ylab = "CSMF", col=alpha(cols[ii], 0.6))
		if(ii > 1) lines(toplot, csmfall[ii, order[i], toplot], type = "l", col=alpha(cols[ii], 0.6))
	}	
}
dev.off()


## --------------------------------------------------------------------##
## Task 3: Cross validation
## --------------------------------------------------------------------##

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

nitr <- 50
for(itr in 1:nitr){
	if(file.exists(paste0("../data/processed/karongaCV", itr, ".rda"))) next
	metric <- array(NA, dim = c(9, 4, 3))
	dimnames(metric)[[2]] <- c("CSMF", "Top 1", "Top 2", "Top 3")
	dimnames(metric)[[1]] <- c("InterVA: prior", "InterVA: train", "Naive Bayes: prior", "Naive Bayes: train", "InSilicoVA: prior", "InSilicoVA: train", "King-Lu", "Tariff", "Gaussian Mixture")
	dimnames(metric)[[3]] <- c("5%", "10%", "20%")
	for(case in 1:3){
		dir <- paste0("../experiments/Karonga2019-", itr, "-", case, "-CV1/Karonga2019-", itr, "-", case, "-CV1_")
		train_sub <- read.csv(paste0("../data/expnew/K_", itr, "_train", case, ".csv"), header = F)
		delta <- read.csv(paste0("../data/expnew/typeK3_delta.csv"), header = F)
		test_sub <- read.csv(paste0("../data/expnew/K_", itr, "_test", case, ".csv"), header = F)
		G <- dim(delta)[1]
		csmf <- rep(1/G, G)
		csmf.train <- csmf
		probbase <- 1 - pnorm(as.matrix(delta))
		P <- dim(probbase)[2]
		causes <- c(test_sub[, 1])
		csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))
		# Latent Gaussian results
		prob <- t(as.matrix(fread(paste0(dir, "assignment_out_mean.txt"))))
		csmf <- as.matrix(fread(paste0(dir, "prob_out.txt")))[, -1]
		sub <- 1:dim(csmf)[2]
		sub <- sub[(length(sub)/2+1):length(sub)]
		sub <- sub[sub%%10 == 0]
		csmf.mean <- apply(csmf[, sub], 1, mean)
	
		metric["Gaussian Mixture",, case] <- getAccuracy(prob, test_sub[, 1], csmf.true, csmf.mean)
		# Using prior information 
		fit.nb <- evalNBprob2(probbase = probbase, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
		# Using training data
		probbase2 <- getProbbase3(cbind(1,train_sub), G, 1:G)
		fit.nb.noprior <- evalNBprob2(probbase = probbase2, training = train_sub, testing = test_sub, G, csmf = csmf.train, csmf.true = csmf.true, samepop = FALSE)
		metric["Naive Bayes: prior",, case] <- fit.nb$fitted.nb
		metric["InterVA: prior",, case] <- fit.nb$fitted.inter
		metric["Naive Bayes: train",, case] <- fit.nb.noprior$fitted.nb
		metric["InterVA: train",, case] <- fit.nb.noprior$fitted.inter
		
		## Other methods
		# test0 <- matrix(NA, dim(test_sub)[1], P)
		# test0[test_sub[,-1] == "Y"] <- 1
		# test0[test_sub[,-1] == "N"] <- 0
		# indic.test<- test0
		# train0 <- matrix(NA, dim(train_sub)[1], P)
		# train0[train_sub[,-1] == "Y"] <- 1
		# train0[train_sub[,-1] == "N"] <- 0
		# indic.train <- train0
		# indic.test[is.na(indic.test)] <- 0
		# indic.train[is.na(indic.train)] <- 0
		# symcount1 <- apply(indic.test, 2, sum)
		# symcount2 <- apply(indic.train, 2, sum)   
		# invar <- union(which(symcount1 %in% c(0, dim(indic.test)[1])), which(symcount2 %in% c(0, dim(indic.train)[1])))
		# indic.test<- indic.test[, -invar]
		# indic.train <- indic.train[, -invar]    
		# colnames(indic.test) <- colnames(indic.train) <- paste0("V", 1:dim(indic.train)[2])
		# indic.train <- cbind(cod = train_sub[, 1], indic.train)
		# indic.test <- cbind(cod = test_sub[, 1], indic.test)
		# p <- dim(indic.test)[2]-1
		# formula <- as.formula(paste0("cbind(V1+...+V", p, ")~cod"))
		# KL <- va(formula, data = list(indic.train, indic.test), nsymp=18) 
		# csmf.KL <- KL$est.CSMF
		# metric["King-Lu", 1, case] <- getAccuracy(prob, test_sub[, 1], csmf.true, csmf.KL)[1]

		test_sub_1 <- cbind(ID = 1:dim(test_sub)[1], test_sub)
		test_sub_1 <- ConvertData(test_sub_1, yesLabel = "Y", noLabel = c("N"), missLabel=".")
		train_sub_1 <- cbind(ID = 1:dim(train_sub)[1], train_sub)
		train_sub_1 <- ConvertData(train_sub_1, yesLabel = "Y", noLabel = c("N", "."), missLabel = NA)

		colnames(test_sub_1) <- colnames(train_sub_1) <- c("ID", "Cause", colnames(probbase))
		colnames(probbase2) <- colnames(probbase)
		fit <- insilico.train2(data = test_sub_1, data.type = "customize", train = train_sub_1, cause = "Cause", causes.table = as.character(1:G), Nsim = 10000, auto.length = FALSE, updateCondProb = FALSE, datacheck=FALSE, burnin = 5000, CondProbNum = probbase, type = "empirical")
		csmf.tmp <- getCSMF(fit)[,1]
		prob.tmp <- getIndivProb(fit)
		metric["InSilicoVA: prior", , case] <- getAccuracy(prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)
		
		fit1 <- insilico.train2(data = test_sub_1, data.type = "customize", train = train_sub_1, cause = "Cause", causes.table = as.character(1:G), Nsim = 10000, auto.length = FALSE, updateCondProb = FALSE, datacheck=FALSE, burnin = 5000, CondProbNum = probbase2, type = "empirical")
		csmf1.tmp <- getCSMF(fit1)[,1]
		prob1.tmp <- getIndivProb(fit1)
		metric["InSilicoVA: train", , case] <- getAccuracy(prob1.tmp, test_sub[, 1], csmf.true, csmf1.tmp)

		fit <- codeVA(data = test_sub_1, data.type = "customize", data.train = train_sub_1, causes.train = "Cause", causes.table = as.character(1:G), model = "Tariff")
		csmf.tmp <- getCSMF(fit)[as.character(1:G)]
		names(csmf.tmp) <- as.character(1:G)
		csmf.tmp[is.na(csmf.tmp)] <- 0
		prob.tmp <- getIndivProb(fit)
		if(dim(prob.tmp)[2] < G){
			add <- matrix(max(prob.tmp)+1, dim(prob.tmp)[1], G - dim(prob.tmp)[2])
			colnames(add) <- which(c(1:G) %in% colnames(prob.tmp) == FALSE)
			prob.tmp <- cbind(prob.tmp, add)
		} 
		prob.tmp <- prob.tmp[, as.character(1:G)]
		metric["Tariff", , case] <- getAccuracy(-prob.tmp, test_sub[, 1], csmf.true, csmf.tmp)
	}
	print(metric)
	save(metric, file = paste0("../data/processed/karongaCV", itr, ".rda"))
}


library(plyr)
library(ggplot2)
library(gridExtra)
metrics.all <- data.frame(matrix(NA, nitr*9*4*3, 5))
colnames(metrics.all) <- c("Method", "Metric", "Value", "Rep", "Type")
prop <- c("Labeled 5%", "Labeled 10%", "Labeled 20%")
counter <- 1
for(itr in 1:50){
	if(!file.exists(paste0("../data/processed/karongaCV", itr, ".rda"))){
		cat(itr, " ")
		next
	}
	load(paste0("../data/processed/karongaCV", itr, ".rda"))
	for(i in 1:3){
			tmp <- data.frame(Method  = rep(rownames(metric[,,i]), 4), 
					  Metric  = rep(colnames(metric[,,i]), each = 9), 
					  Value = as.numeric(metric[,,i]), 
					  Rep = itr, 
					  Type = prop[i],
					  stringsAsFactors=FALSE)
			metrics.all[counter : (counter + 9*4 - 1), ] <- tmp
			counter <- counter + 9*4 
	}
}

metrics.all$Metric <-factor(metrics.all$Metric, levels= c("CSMF", "Top 1", "Top 2", "Top 3"))
metrics.all$Metric <- revalue(metrics.all$Metric, c("CSMF"="CSMF Accuracy", "Top 1" = "Top Cause Acc", "Top 2" = "Top 2 Cause Acc", "Top 3" = "Top 3 Cause Acc"))
metrics.all$Method2 <- revalue(metrics.all$Method, c("InterVA: prior"="InterVA",  "InterVA: train"="InterVA", "Naive Bayes: prior" = "Naive Bayes", "Naive Bayes: train" = "Naive Bayes", "InSilicoVA: prior"="InSilicoVA",  "InSilicoVA: train"="InSilicoVA"))
metrics.all$group <- revalue(metrics.all$Method, c("InterVA: prior"="prior",  "InterVA: train"="training", "Naive Bayes: prior" = "prior", "Naive Bayes: train" = "training", "InSilicoVA: prior"="prior",  "InSilicoVA: train"="training", "King-Lu" = "training", "Tariff" = "training", "Gaussian Mixture" = "Both"))
metrics.all$Type <- factor(metrics.all$Type, levels = prop)
mets <- c("Tariff",  "InterVA" , "Naive Bayes", "InSilicoVA", "King-Lu", "Gaussian Mixture") 
metrics.all$Method2 <- factor(metrics.all$Method2, levels = mets) 
cbPalette <- c('#377eb8','#4daf4a','#984ea3','#ff7f00','#e41a1c', '#a65628')
g1 <- ggplot(subset(metrics.all, Metric == "CSMF Accuracy" & Method %in% c( "King-Lu") ==FALSE), aes(x = Method2, y = Value, color = Method2, group = Method, fill = group)) + facet_wrap(~Type) + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("Accuracy") + scale_color_manual(values=cbPalette) + scale_fill_manual(values = c("white", "white", "gray80")) + xlab("")


g2 <- ggplot(subset(metrics.all, Metric == "Top Cause Acc" & Method %in% c("King-Lu") ==FALSE), aes(x = Method2, y = Value, color = Method2, group = Method, fill = group)) + facet_wrap(~Type) + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("Accuracy") + scale_color_manual(values=cbPalette) + scale_fill_manual(values = c("white", "white", "gray80")) + xlab("")

 
g3 <- ggplot(subset(metrics.all, Metric == "Top 3 Cause Acc" & Method %in% c("King-Lu") ==FALSE), aes(x = Method2, y = Value, color = Method2, group = Method, fill = group)) + facet_wrap(~Type) + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("Accuracy") + scale_color_manual(values=cbPalette) + scale_fill_manual(values = c("white", "white", "gray80")) + xlab("")


pdf("../figures/Karonga_compare_box-CSMF.pdf", width = 9, height=5)
print(g1)
dev.off()
 
pdf("../figures/Karonga_compare_box-top1.pdf", width = 9, height=5)
print(g2)
dev.off()
 
pdf("../figures/Karonga_compare_box-top3.pdf", width = 9, height=5)
print(g3)
dev.off()


