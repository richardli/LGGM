source("FanEstimator.r")
csmfacc <- function(csmf, csmf.fit){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}
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

#######################################################
## PHMRC 1000-1000
## random sampling 
## 19B : non-informative prior, Dirichlet
## 22B : informative prior, Dirichlet
#######################################################
library(ggplot2)
library(reshape2)
library(plyr)
csmfacc <- function(csmf.fit, csmf){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}

metrics <- NULL
training <- NULL
reps <- 1:50
for(train in c(0)){
	for(rep in reps){
		filename <- paste0("rdaVA/201802/PHMRC9/newPS9-", train, "-", rep, "-20F.rda")
		if(!file.exists(filename)){
			cat("?")
			next
		}
		load(filename)
		csmf.true <- out$fitted.nb$csmf.true
		csmf1 <- apply(out$csmf.s1, 1, mean)
		csmf2 <- apply(out$pnb_integral.s1, 2, mean)
		csmf3 <- apply(out$csmf, 1, mean) 
		csmf4 <- apply(out$pnb_integral, 2, mean)
		csmf5 <- out$fitted.nb$csmf.nb
		csmf6 <- out$fitted.nb.noprior$csmf.nb	
		csmf7 <- out$fitted.nb$csmf.inter
		out$metric[1, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[2, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[3, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[4, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[5, 1] <- csmfacc(csmf5, csmf.true)
		out$metric[6, 1] <- csmfacc(csmf6, csmf.true)
		out$metric[7, 1] <- csmfacc(csmf7, csmf.true)
		row <- as.numeric(out$metric[, c(1,2,4)])
		metrics <- rbind(metrics, row)
		training <- c(training, train)
		cat(".")
	}
}
metrics <- data.frame(metrics)
types <- c("Proposed0S1", "ProposedS1", "Proposed0", "Proposed","NaiveBayesPrior", "NaiveBayesTraining", "InterVA")
csmf <- metrics[, 1:7]
acc1 <- metrics[, 8:14]
acc3 <- metrics[, 15:21]
colnames(csmf) <- colnames(acc1)<- colnames(acc3) <- types
toplot <- data.frame(rbind(csmf, acc1, acc3))
toplot$type <- c(rep("CSMF", dim(csmf)[1]),
				 rep("Top Cause Acc", dim(acc1)[1]),
				 rep("Top 3 Cause Acc", dim(acc3)[1]))

metrics <- NULL
training <- NULL
reps <- 1:50
for(train in c(0)){
	for(rep in reps){
		filename <- paste0("rdaVA/201802/PHMRC9/newPS9-", train, "-", rep, "-20Enotrain.rda")
		if(!file.exists(filename)){
			cat("?")
			next
		}
		load(filename)
		csmf.true <- out$fitted.nb$csmf.true
		csmf1 <- apply(out$csmf.s1, 1, mean)
		csmf2 <- apply(out$pnb_integral.s1, 2, mean)
		csmf3 <- apply(out$csmf, 1, mean) 
		csmf4 <- apply(out$pnb_integral, 2, mean)
		csmf5 <- out$fitted.nb$csmf.nb
		# csmf6 <- out$fitted.nb.noprior$csmf.nb	
		csmf7 <- out$fitted.nb$csmf.inter
		out$metric[1, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[2, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[3, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[4, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[5, 1] <- csmfacc(csmf5, csmf.true)
		# out$metric[6, 1] <- csmfacc(csmf6, csmf.true)
		out$metric[7, 1] <- csmfacc(csmf7, csmf.true)
		row <- as.numeric(out$metric[, c(1,2,4)])
		metrics <- rbind(metrics, row)
		training <- c(training, train)
		cat(".")
	}
}
metrics <- data.frame(metrics)
types <- c("Proposed0S1", "ProposedS1", "Proposed0", "Proposed","NaiveBayesPrior", "NaiveBayesTraining", "InterVA")
csmf <- metrics[, 1:7]
acc1 <- metrics[, 8:14]
acc3 <- metrics[, 15:21]
colnames(csmf) <- colnames(acc1)<- colnames(acc3) <- types
toplot2 <- data.frame(rbind(csmf, acc1, acc3))
toplot2$type <- c(rep("CSMF", dim(csmf)[1]),
				 rep("Top Cause Acc", dim(acc1)[1]),
				 rep("Top 3 Cause Acc", dim(acc3)[1]))
colnames(toplot2)[1:4] <- paste0("noTrain", colnames(toplot2)[1:4])
toplot <- cbind(toplot, toplot2[, 1:4])
# toplot$train <- factor(toplot$train, levels = c("w/o training data", "w/i training data"))

# pdf("../Figures/2018/PHMRC_compare.pdf", width = 7.5, height=3)
# g <- ggplot(aes(x = NaiveBayesPrior, y = Proposed), data = toplot)
# g <- g + geom_point(data=subset(toplot, type == "Top 3 Cause Acc"))
# g <- g + xlim(range(subset(toplot, type == "Top 3 Cause Acc")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + ylim(range(subset(toplot, type == "Top 3 Cause Acc")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + geom_abline(intercept=0, slope=1, linetype="dashed", color="red")
# g <- g + theme_bw()
# g <- g + xlab("Naive Bayes classifier") + ylab("Proposed SS prior")
# g <- g + ggtitle("Classification accuracy for PHMRC data")
# g <- g + facet_wrap(~train)
# g
# dev.off()

# pdf("../Figures/2018/PHMRC_compare_csmf.pdf", width = 7.5, height=3)
# g <- ggplot(aes(x = NaiveBayesPrior, y = Proposed), data = toplot)
# g <- g + geom_point(data=subset(toplot, type == "CSMF"))
# g <- g + xlim(range(subset(toplot, type == "CSMF")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + ylim(range(subset(toplot, type == "CSMF")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + geom_abline(intercept=0, slope=1, linetype="dashed", color="red")
# g <- g + theme_bw()
# g <- g + xlab("Naive Bayes classifier") + ylab("Proposed SS prior")
# g <- g + ggtitle("CSMF accuracy for PHMRC data")
# g <- g + facet_wrap(~train)
# g
# dev.off()


toplot.long <- melt(toplot)
toplot.long$variable <- revalue(toplot.long$variable, c("Proposed0S1"="none",
	 "ProposedS1"="GM w/i training", 
	 "Proposed"="none",
	"Proposed0"="none",
	"noTrainProposed0S1"="none",
	 "noTrainProposedS1"="GM w/o training", 
	 "noTrainProposed"="none",
	"noTrainProposed0"="none",
	"NaiveBayesPrior" = "Naive Bayes",
	"NaiveBayesTraining" = "none"
	))
toplot.long <- subset(toplot.long, variable != "none")
toplot.long$variable <- factor(toplot.long$variable, levels = c("InterVA", "Naive Bayes", 
		"NB: training", 
	"GM w/o training", "GM w/i training"))
toplot.long$type <- factor(toplot.long$type, levels = c("Top Cause Acc", "Top 3 Cause Acc", "CSMF"))
# combine with and without training
# select <- intersect(which(toplot.long$train == "CSMF w/i training"), which(toplot.long$variable == "SS: One-stage"))


pdf("../Figures/2018/PHMRC_compare_box.pdf", width = 9, height=4.5)
g <- ggplot(aes(x =variable, y = value, fill=variable), data = toplot.long)
g <- g + geom_boxplot()
# g <- g + facet_grid(type~train, scales="free_y")
g <- g + facet_wrap(~type, scales="free_y")
g <- g + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))+ guides(fill=guide_legend("none"))
g <- g + xlab("") + ylab("Accuracy")
g <- g + ggtitle("")
g
dev.off()


# Example CSMF plot
pnb <- t(apply(out$pnb_integral.s1, 1, function(x, csmf){(x/csmf) / sum(x / csmf)}, csmf = apply(out$csmf, 1, mean)))
pnb.draw <- array(NA, dim = c(dim(pnb)[1], dim(pnb)[2], dim(out$csmf)[2]))
for(itr in 1: dim(out$csmf)[2]){
	pnb.draw[,,itr] <- t(t(pnb) * out$csmf.s1[,itr])
	pnb.draw[,,itr] <- t(apply(pnb.draw[,,itr], 1, function(x){x/sum(x)}))
}
csmf2 <- apply(pnb.draw, c(2,3), mean)
csmf.plot <- t(apply(out$csmf.s1, 1, quantile, c(.025, .5, .975))) 
csmf.plot2 <- t(apply(csmf2, 1, quantile, c(.025, .5, .975))) 
colnames(csmf.plot) <- c("lower", "med", "upper")
colnames(csmf.plot2) <- c("lower", "med", "upper")
csmf.plot <- data.frame(csmf.plot)
csmf.plot2 <- data.frame(csmf.plot2)
csmf.plot$sample <- apply(out$pnb_integral.s1, 2, mean)
csmf.plot2$sample <- apply(out$pnb_integral.s1, 2, mean)
csmf.plot$cause <- as.character(1:dim(csmf.plot)[1])
csmf.plot2$cause <- as.character(1:dim(csmf.plot)[1])
csmf.plot$type <- "draw"
csmf.plot2$type <- "plug-in"
csmf.plot <- rbind(csmf.plot, csmf.plot2)
csmf.plot$cause <- factor(csmf.plot$cause, levels = csmf.plot2$cause[order(csmf.plot2$sample)])
g <- ggplot(aes(x = cause, y = med, ymin=lower, ymax=upper, color = type), data = csmf.plot)
g <- g + geom_point() + geom_errorbar()
g <- g + geom_point(aes(x = cause, y = sample), color = "blue")
# g <- g + geom_point(aes(x = cause, y = truth), color = "gold")
g <- g + coord_flip()
g


 


#######################################################
## Karonga 16 causes CV
## 2007 and before: probbase
## random sampling: 5%, 10% and 20% training
#######################################################
library(ggplot2)
library(reshape2)
library(plyr)
csmfacc <- function(csmf.fit, csmf){ 1-sum(abs(csmf.fit - csmf))/2/(1-min(csmf))}

metrics <- NULL
training <- NULL
reps <- 1:20
for(train in c(0:3)){
	for(rep in reps){
		filename <- paste0("rdaVA/201802/newK9F/newK9-", rep, "-", train, "-20F.rda")
		if(train==0){
			filename <- paste0("rdaVA/201802/newK9F/newK9-", rep, "-", train, "-20Fnotrain.rda")
		}
		if(!file.exists(filename)) next

		load(filename)
		load(filename)
		csmf.true <- out$fitted.nb$csmf.true
		csmf1 <- apply(out$csmf.s1, 1, mean)
		csmf2 <- apply(out$pnb_integral.s1, 2, mean)
		csmf3 <- apply(out$csmf, 1, mean) 
		csmf4 <- apply(out$pnb_integral, 2, mean)
		csmf5 <- out$fitted.nb$csmf.nb
		csmf6 <- out$fitted.nb.noprior$csmf.nb	
		csmf7 <- out$fitted.nb$csmf.inter
		out$metric[1, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[2, 1] <- csmfacc(csmf1, csmf.true)
		out$metric[3, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[4, 1] <- csmfacc(csmf3, csmf.true)
		out$metric[5, 1] <- csmfacc(csmf5, csmf.true)
		out$metric[6, 1] <- csmfacc(csmf6, csmf.true)
		out$metric[7, 1] <- csmfacc(csmf7, csmf.true)
		if(train == 0) out$metric[6,]<- NA
		row <- as.numeric(out$metric[, c(1,2)])
		metrics <- rbind(metrics, row)
		training <- c(training, c(0, 0.05, 0.1, 0.2)[train+1])
		cat(".")
	}
}
metrics <- data.frame(metrics)
types <- c("Proposed0S1", "ProposedS1", "Proposed0", "Proposed","NaiveBayesPrior", "NaiveBayesTraining", "InterVA")
csmf <- metrics[, 1:7]
acc <- metrics[, 8:14]
colnames(csmf) <- colnames(acc) <- types

toplot <- data.frame(rbind(csmf, acc))
toplot$type <- c(rep("CSMF", length(training)),
				 rep("Acc", length(training)))
toplot$training <- rep(paste0("Training: ", training * 100, "%"), 2)
toplot$training <- factor(toplot$training, levels = c("Training: 0%", "Training: 5%","Training: 10%", "Training: 20%"))

# pdf("../Figures/2018/Karonga_compare.pdf", width = 7.5, height=7.5)
# g <- ggplot(aes(x = NaiveBayesPrior, y = Proposed), data = toplot)
# g <- g + geom_point(data=subset(toplot, type == "Acc"))
# g <- g + xlim(range(subset(toplot, type == "Acc")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + ylim(range(subset(toplot, type == "Acc")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + geom_abline(intercept=0, slope=1, linetype="dashed", color="red")
# g <- g + facet_wrap(~training)
# g <- g + theme_bw()
# g <- g + xlab("Naive Bayes classifier") + ylab("Proposed SS prior")
# g <- g + ggtitle("Classification accuracy for Karonga data")
# g
# dev.off()

# pdf("../Figures/2018/Karonga_compare_csmf.pdf", width = 7.5, height=7.5)
# g <- ggplot(aes(x = NaiveBayesPrior, y = Proposed), data = toplot)
# g <- g + geom_point(data=subset(toplot, type == "CSMF"))
# g <- g + xlim(range(subset(toplot, type == "CSMF")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + ylim(range(subset(toplot, type == "CSMF")[, c( "NaiveBayesPrior","Proposed")]))
# g <- g + geom_abline(intercept=0, slope=1, linetype="dashed", color="red")
# g <- g + facet_wrap(~training)
# g <- g + theme_bw()
# g <- g + xlab("Naive Bayes classifier") + ylab("Proposed SS prior")
# g <- g + ggtitle("CSMF accuracy for Karonga data")
# g
# dev.off()


toplot.long <- melt(toplot)
toplot.long$variable <- revalue(toplot.long$variable, c("Proposed0"="none",
	 "ProposedS1"="GM", 
	 "Proposed"="none",
	"Proposed0S1"="none",
	"NaiveBayesPrior" = "NB: prior",
	"NaiveBayesTraining" = "NB: training"
	))
toplot.long <- subset(toplot.long, variable != "none")
toplot.long$variable <- factor(toplot.long$variable, levels = c("InterVA", "NB: prior", "NB: training", "GM" ))
pdf("../Figures/2018/Karonga_compare_box.pdf", width = 9, height=6)
g <- ggplot(aes(x =variable, y = value, fill=variable), data = subset(toplot.long, type == "Acc"))
g <- g + geom_boxplot()
g <- g + facet_wrap(~training, nrow = 1)
g <- g + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))+ guides(fill=guide_legend("none"))
g <- g + xlab("") + ylab("Classification accuracy")
g <- g + ggtitle("Classification accuracy for Karonga data")
g
dev.off()
pdf("../Figures/2018/Karonga_compare_csmf_box.pdf", width = 9, height=6)
g <- ggplot(aes(x =variable, y = value, fill=variable), data = subset(toplot.long, type == "CSMF"))
g <- g + geom_boxplot()
g <- g + facet_wrap(~training, nrow = 1)
g <- g + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))+ guides(fill=guide_legend("none"))
g <- g + xlab("") + ylab("Classification accuracy")
g <- g + ggtitle("CSMF accuracy for Karonga data")
# g + ylim(0.5, 0.9)
g
dev.off()


# pdf("../Figures/2018/Karonga_compare_box_slides.pdf", width = 9, height=6)
# toplot.long$variable <- revalue(toplot.long$variable, 
# 	c("Proposed SS prior" = "Proposed\nSS prior",
# 	  "Naive Bayes: using prior"="Naive Bayes\nusing prior",
# 	  "Naive Bayes: using training data"="Naive Bayes\nusing training"))
# g <- ggplot(aes(x =variable, y = value, fill=variable), data = subset(toplot.long, type == "Acc"))
# g <- g + geom_boxplot()
# g <- g + facet_wrap(~training)
# g <- g + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))+ guides(fill=guide_legend("none"))
# g <- g + xlab("") + ylab("Classification accuracy")
# g <- g + ggtitle("Classification accuracy for Karonga data")
# g <- g + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15))
# g
# dev.off()
# pdf("../Figures/2018/Karonga_compare_csmf_box_slides.pdf", width = 9, height=6)
# g <- ggplot(aes(x =variable, y = value, fill=variable), data = subset(toplot.long, type == "CSMF"))
# g <- g + geom_boxplot()
# g <- g + facet_wrap(~training)
# g <- g + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))+ guides(fill=guide_legend("none"))
# g <- g + xlab("") + ylab("Classification accuracy")
# g <- g + ggtitle("CSMF accuracy for Karonga data")
# g <- g + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 15))
# g
# dev.off()




#######################################################
# 20F contains training data
# maybe change to 21F, no training
library(corrplot)
library(ggrepel)
filename="rdaVA/201802/newK9F/newK9-0-0-21Fnotrain.rda"
load(filename)
out0 <- out #without training data
filename="rdaVA/201802/newK9F/newK9-0-0-20F.rda"
load(filename) #with training data

csmfs <- out$csmf.s1
csmf.plot <- t(apply(csmfs,1, quantile, c(.025, .5, .975)))
colnames(csmf.plot) <- c("lower", "median", "upper")
csmf.plot <- data.frame(csmf.plot)
csmf.plot$sample <- apply(out$pnb_integral, 2, mean)
csmf.plot$nb <- out$fitted.nb$csmf.nb
csmf.plot$inter <- out$fitted.nb$csmf.inter
csmf.plot$truth <- out$fitted.nb$csmf.true

csmf.plot$cause <- c("TB/AIDS",                                       
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


csmf.plot2 <- melt(csmf.plot, id = c("lower", "upper", "truth", "cause"))
# add in the no training data version
tmp <- out0$csmf.s1
tmp <- t(apply(tmp,1, quantile, c(.025, .5, .975)))
colnames(tmp) <- c("lower", "value", "upper")
tmp <- data.frame(tmp)
tmp$truth <- out0$fitted.nb$csmf.true
tmp$cause <- csmf.plot$cause
tmp$variable <- "notrain"
csmf.plot2 <- rbind(csmf.plot2, tmp)
colnames(csmf.plot2)[colnames(csmf.plot2) == "variable"] <- "Method"
csmf.plot2 <- csmf.plot2[-which(csmf.plot2$Method == "sample"), ]
csmf.plot2$cause <- as.character(csmf.plot2$cause)
csmf.plot2$lower[csmf.plot2$Method %in% c("nb", "inter")] <- NA
csmf.plot2$upper[csmf.plot2$Method %in% c("nb", "inter")] <- NA
csmf.plot2$Method <- factor(csmf.plot2$Method, 
	levels = c("inter", "nb", "notrain", "median"))
csmf.plot2$Method <- revalue(csmf.plot2$Method, c("median"="Gaussian Mixture", "notrain"="Proposed (w/o training)", "nb"="Naive Bayes", "inter"="InterVA"))
lim <- range(c(csmf.plot2$value), na.rm=T)
csmf.plot2$cause[csmf.plot2$truth < 0.05] <- ""

pdf("../Figures/2018/Karonga_csmf.pdf", width = 10, height=3.5)
g <- ggplot(data = subset(csmf.plot2, Method != "Proposed (w/o training)"))
g <- g + geom_errorbar(aes(x=truth, y=value, ymin=lower, ymax=upper), alpha = 0.8, color = "#377eb8")
g <- g + geom_point(aes(x=truth, y=value), col="#d7301f", size = 1.8)
# g <- g + geom_text(aes(x=truth, y=value, label=cause), col="red", size = 2.5, nudge_y = 0.00, nudge_x = 0.01)
g <- g + geom_text_repel(aes(truth, value, label=cause), color = "#d73027", size = 2.5, segment.alpha=0, force = 1.2)
g <- g + facet_wrap(~Method)
g <- g + geom_abline(slope=1,intercept=0, linetype="longdash")
g <- g + xlim(lim) + ylim(lim) + theme_bw()
g <- g + xlab("True CSMF") + ylab("Estimated CSMF")
g
dev.off()

library(openVA)
library(igraph)
library(arcdiagram)
library(corrplot)
library(xtable)
train_sub <- read.csv(paste0("../data/expnew/K_train0.csv"), header = F)
test_sub <- read.csv(paste0("../data/expnew/K_test0.csv"), header = F)
G <- 16
causes <- c(test_sub[, 1])
csmf.true <- out$fitted.nb$csmf.true
csmf1 <- apply(out$csmf.s1, 1, mean)
csmf2 <- apply(out$pnb_integral.s1, 2, mean)
csmf3 <- apply(out$csmf, 1, mean) 
csmf4 <- apply(out$pnb_integral, 2, mean)
csmf5 <- out$fitted.nb$csmf.nb
csmf6 <- out$fitted.nb.noprior$csmf.nb	
csmf7 <- out$fitted.nb$csmf.inter
out$metric[1, 1] <- csmfacc(csmf1, csmf.true)
out$metric[2, 1] <- csmfacc(csmf1, csmf.true)
out$metric[3, 1] <- csmfacc(csmf3, csmf.true)
out$metric[4, 1] <- csmfacc(csmf3, csmf.true)
out$metric[5, 1] <- csmfacc(csmf5, csmf.true)
out$metric[6, 1] <- csmfacc(csmf6, csmf.true)
out$metric[7, 1] <- csmfacc(csmf7, csmf.true)
csmf1a <- apply(out0$csmf.s1, 1, mean) 
out0$metric[2, 1] <- csmfacc(csmf1a, csmf.true)
tab <- rbind(out$metric[c(7, 5), ], 
			 out0$metric[2, ], 
			 out$metric[2, ])
rownames(tab) <- c("InterVA", "Naive Bayes", "GM w/o training", "GM w/i training")
xtable(tab)


inclusion <-  out$inclusion
corrs <- invcorrs <- structs <- NULL
corrs[[1]] <- out$corr.mean
invcorrs[[1]] <- out$prec.mean
structs[[1]] <- as.matrix(inclusion)
load("../data/expnew/sympsK.rda")
data(probbase3)
symps[symps == "skin_les"] <- "sk_les"
sympstext <- probbase3[match(symps, probbase3[, 2]), 3]
structures <- as.matrix(read.csv("../data/expnew/typeK_structure.csv", header = F))
colnames(structures) <- rownames(structures) <- symps

invcorrs.sp <- NULL
plot.out <- NULL
exist <- NULL
for(i in 1:length(corrs)){
	invcorrs.sp[[i]] <- invcorrs[[i]] * (structs[[i]] > 1499/2)
	rownames(invcorrs.sp[[i]]) <- colnames(invcorrs.sp[[i]]) <- symps
	if(is.null(exist)){
		exist <- rep(0, dim(invcorrs[[i]])[1])
	}
	tmp <- invcorrs.sp[[i]]
	diag(tmp) <- 0
	exist <- exist + as.numeric(apply(tmp, 2, function(x){
		sum(abs(x)>0,na.rm=T)>0
		}))
}

adj0<- matrix(0, dim(invcorrs[[1]])[1], dim(invcorrs[[1]])[1])
mean <- matrix(0, dim(invcorrs[[1]])[1], dim(invcorrs[[1]])[1])
for(i in 1:length(invcorrs)){
	tmp <- abs(invcorrs.sp[[i]])
	tmp[tmp != 0] <- 1
	adj0 <- adj0 + tmp
	mean <- mean + cov2cor(invcorrs.sp[[i]]) / length(invcorrs)
}

pdf("../figures/2018/corrfit.pdf", width = 5.5, height = 5.5)
temp <- corrs[[1]]
colnames(temp) <- rep("", dim(temp)[1])
rownames(temp) <- rep("", dim(temp)[1])
corrplot(temp, method="color", 
	title = paste0("Correlation matrix"), bg = "gray", diag=T, tl.cex = 0.5, tl.srt=45, is.corr=T, mar = c(1,1,2,1)-0.9, cl.pos='r')	
dev.off()

pdf("../figures/2018/invcorrfit.pdf", width = 5.5, height = 5.5)
temp <- invcorrs[[1]]
colnames(temp) <- rep("", dim(temp)[1])
rownames(temp) <- rep("", dim(temp)[1])
corrplot(temp, method="color", 
	title = paste0("Inverse Correlation matrix"), bg = "gray", diag=T, tl.cex = 0.5, tl.srt=45, is.corr=F, mar = c(1,1,2,1)-0.9, cl.pos='r')	
dev.off()

pdf("../figures/2018/structfit.pdf", width = 5, height = 5)
temp <- as.matrix(structs[[1]]) / 1499
# temp[temp < 1499/2] <- 0
# temp[temp != 0] <- temp[temp != 0] 
temp[structures != 0] <- 0
diag(temp) <- 0
colnames(temp) <- rep("", dim(temp)[1])
rownames(temp) <- rep("", dim(temp)[1])
temp2 <- structures 
diag(temp2) <- 1
corrplot(temp, method="color", 
	title = paste0("Posterior selection probability"), bg = "gray", diag=T, tl.cex = 0.5, tl.srt=45, is.corr=F, mar = c(1,1,2,1)-0.9, cl.lim = c(0, 1), 
		 p.mat = temp2, sig.level = 0.5, insig = c("pch"), pch = 15 , pch.cex = 0.75, pch.col = "orange")	
dev.off()
 
adj <- adj0
rownames(mean) <- colnames(mean) <- symps
rownames(structs[[1]]) <- colnames(structs[[1]]) <- symps
# adj[adj < 0.5 * length(invcorrs)] <- 0
diag(structures) <- 0
adj <- adj * (1-structures)
diag(adj) <- 0
graph=graph.adjacency(adj, mode="undirected",diag=FALSE)
edgelist = get.edgelist(graph)
edgelist <- unique(edgelist)
values <- rep(0, dim(edgelist)[1])
freqs <- rep(0, dim(edgelist)[1])
for(i in 1:length(values)){
	values[i] <- mean[edgelist[i, 1], edgelist[i, 2]]
	freqs[i] <- structs[[1]][edgelist[i, 1], edgelist[i, 2]] / 1499
}
edges_text <- edgelist
for(i in 1:2) edges_text[, i] <- sympstext[match(edges_text[, i], symps)]
# write.csv(edges_text, "../Figures/2018/correlated_symps.csv")

e5 <- cbind(round(freqs, 2), edges_text, round(-values, 2))
e5 <- e5[order(freqs, decreasing = T), ]
freqs.ordered <- freqs[order(freqs, decreasing = T)]
print(xtable(e5[1:20, ]), include.rownames=FALSE)
print(xtable(e5), include.rownames=FALSE)
sum(freqs.ordered > 0.5)



############################################################
## Karonga 16 causes Read data
############################################################
raw1 <- read.csv("../data/raw/alpha8_insilico_input_karonga.csv")
raw1 <- raw1[, c(1:247, 249, 256-1, 260-1, 264-1)]
data1 <- raw1[, c(3:247)] # 1900 
year1 <- raw1[, 248]
# remove external symptoms
removesymp <- c("injury", "traffic", "o_trans", "fall", "drown", "fire", "assault", "vemon", "force", "poison", "inflict", "suicide")
data1 <- data1[, !(colnames(data1) %in% removesymp)]
################################################################
# Remove those without physician codes
################################################################
cod <- as.character(raw1[, 249])
disagree = which(as.character(raw1[, 249]) != as.character(raw1[, 250])) 
cod[disagree] <- as.character(raw1[disagree, 251])
causes.text <- c("TB/AIDS",                                      
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
exist = which(cod  %in% causes.text)
data1 <- data1[exist, ]  # 1900 -> 1590
cod1 <- cod[exist]
year1 <- year1[exist]

year2 <- year1
year2[year2 <= 2007] <- 0
tab <- as.matrix(table(year2, cod1))
tab0 <- table(cod1)
tab0[which(names(tab0) == "Cause of death unknown")] <- -1
order <- rev(order(tab0))
tab1 <- apply(tab, 1, function(x){x/sum(x)})
tab1 <- tab1[order,]
tab2 <- t(tab)[order, ]
colnames(tab1)[1] <- "<2008"

tab1s <- tab1
tab2s <- tab2
pdf("../Figures/2018/Karonga_summary.pdf", width = 7, height=6)
corrplot(tab1s, is.corr = FALSE, method="color", 
	     p.mat = tab2s, insig = "p-value", sig.level = -1,
	     pch.col="grey", tl.srt = 60, cl.pos="r", cl.lim = c(0, 0.42), cl.cex=.7,
	     mar=c(0.5, 0.1, 1.1, 1.1))
dev.off()

causes.text <- rownames(tab2s)
