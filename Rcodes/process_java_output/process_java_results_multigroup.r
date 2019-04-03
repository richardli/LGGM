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

Cov <- "SSSL"
for(Case in c( "B", "A")){
	for(n.index in 1:3){
		n <- c(0, 100, 200)[n.index]
		for(miss in c("0.0", "0.2", "0.5")){
			p <- 50
			pre0 <- paste0("newCase", Case, "Random",n, Cov, "N", n+800, "P", p, "Miss", miss)
			if(file.exists(paste0("../data/processed/", pre0, "-metrics1.rda"))) next
			files <- list.files(paste0("../experiments/", pre0, "/"))
			files <- files[grep("prob_out.txt", files)]
			files <- files[-grep("s1", files)]
			reps <- gsub("_prob_out.txt", "", gsub(paste0(pre0, "Rep"), "", files))
			allout <- array(NA, c(length(reps), 6, 10))
			if(file.exists(paste0("../data/processed/", pre0, "-metrics1.rda"))) next

			metric <- matrix(NA, 12*length(reps), 7)
			colnames(metric) <- c("Case", "Training", "Method", "Metric", "Value", "Miss", "Seed")
			counter <- 1
			print(length(reps))
			for(ii in 1:length(reps)){
				rep <- reps[ii]
				prefix <- paste0("../experiments/", pre0, "/", pre0, "Rep", rep)
				X <- as.matrix(fread(paste0(prefix, "_X.txt")))
				X[,1] <- X[,1] + 1
				X[X < -1e10] <- NA
				
				type <- rep(1, dim(X)[2] - 1)
				for(i in 2:dim(X)[2]){
					## todo: handle categorical case
					if(sum(!(X[,i] %in% c(0, 1, NA)), na.rm=T)>0){
						type[i-1] <- 0
					}
				}
				delta <- as.matrix(fread(paste0(prefix, "_mean_out_truth.txt")))
				G <- dim(delta)[1]
				csmf <- rep(1/G, G)
				csmf.train <- csmf
				probbase <- 1 - pnorm(as.matrix(delta))
				P <- dim(probbase)[2]
				causes <- c(X[, 1]) 
				csmf.true <- as.numeric((table(c(1:G, causes)) - 1)/length(causes))
				# Latent Gaussian results
				prob <- t(as.matrix(fread(paste0(prefix, "_assignment_out_mean.txt"))))
				csmf <- as.matrix(fread(paste0(prefix, "_prob_out.txt")))[, -1]
				sub <- 1:dim(csmf)[2]
				sub <- sub[(length(sub)/2+1):length(sub)]
				# sub <- sub[sub%%10 == 0]
				csmf.mean <- apply(csmf[, sub], 1, mean)
			
				# Using prior information 
				Ntest <- 800
				out <- rbind(
				naivebayes_eval(type = type, 
					train = X[-(1:Ntest), -1], 
					test = X[1:Ntest, -1], 
					train.Y = causes[-(1:Ntest)], 
					test.Y = causes[1:Ntest], 
					csmf = csmf.true,  
					mean.true = -delta), 
				getAccuracy(prob, X[1:Ntest, 1], csmf.true, csmf.mean))

				metric[counter : (counter + 11), "Case"] <- Case
				metric[counter : (counter + 11), "Training"] <- n
				metric[counter : (counter + 11), "Miss"] <- miss
				metric[counter : (counter + 11), "Seed"] <- rep
				metric[counter : (counter + 11), "Method"] <- rep(c("Naive Bayes", "InterVA", "Gaussian Mixture"), 4)
				metric[counter : (counter + 11), "Metric"] <- rep(colnames(out), each = 3)
				metric[counter : (counter + 11), "Value"] <- as.numeric(out)
				counter <- counter + 12
				print(out)
				print(ii)
			} 
			save(metric, file = paste0("../data/processed/", pre0, "-metrics1.rda"))	
		}
	}
}


library(ggplot2)
library(plyr)
metric.all <- NULL
Cov <- "SSSL"
for(Case in c( "B", "A")){
	for(n.index in 1:3){
		n <- c(0, 100, 200)[n.index]
		for(miss in c("0.0", "0.2", "0.5")){
			p <- 50
			pre0 <- paste0("newCase", Case, "Random",n, Cov, "N", n+800, "P", p, "Miss", miss)
			load(paste0("../data/processed/", pre0, "-metrics1.rda"))
			metric.all <- rbind(metric.all, metric)
		}
	}
}
metric.all <- data.frame(metric.all)
metric.all$Case <- revalue(metric.all$Case, c("A" = "Correct Prior", "B" = "Misspecified Prior"))
metric.all$Method <- factor(metric.all$Method, levels = c("InterVA", "Naive Bayes", "Gaussian Mixture"))
metric.all$Miss <- factor(metric.all$Miss, levels = c("0.0", "0.2", "0.5"))
metric.all$Training <- revalue(as.character(metric.all$Training), c("0" = "N.train = 0", "100" = "N.train = 100", "200" = "N.train = 200"))
metric.all$Miss <- factor(metric.all$Miss, levels = c("0.0", "0.2", "0.5"))
metric.all$Miss <- revalue(as.character(metric.all$Miss), c("0.0" = "Missing Data: 0%", "0.2" = "Missing Data: 20%", "0.5" = "Missing Data: 50%"))

metric.all1 <- subset(metric.all, Method == "Gaussian Mixture")
metric.all2 <- subset(metric.all, Method != "Gaussian Mixture")
metric.all2$Training[metric.all2$Training != "N.train = 200"] <- NA
metric.all <- rbind(metric.all1, metric.all2)
metric.all$Method2 <- paste(metric.all$Method, metric.all$Training, sep = ": ")
table(metric.all$Method2, metric.all$Training)
metric.all$Method2 <- revalue(metric.all$Method2, c("InterVA: N.train = 200" = "InterVA", "Naive Bayes: N.train = 200" = "Naive Bayes"))
metric.all <- metric.all[!is.na(metric.all$Training), ]
metric.all$Method2 <- factor(metric.all$Method2, levels = c("InterVA", "Naive Bayes", paste("Gaussian Mixture: N.train =", c(0, 100, 200))))
metric.all$Value <- as.numeric(as.character(metric.all$Value))

cbPalette <- c('#377eb8','#4daf4a','#984ea3','#ff7f00','#e41a1c', '#a65628')[-1]

pdf("../figures/simCSMF.pdf", width = 9, height=7)
g <- ggplot(data = subset(metric.all, Metric %in% c("CSMF") ), aes(x = Method2, y = Value, color = Method2)) + facet_grid(Case~Miss, scales = "free_x") + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("CSMF Accuracy") + xlab("")+scale_color_manual(values=cbPalette)
print(g)
dev.off()
 
pdf("../figures/simTop1.pdf", width = 9, height=7)
g <- ggplot(data = subset(metric.all, Metric %in% c("top1") ), aes(x = Method2, y = Value, color = Method2)) + facet_grid(Case~Miss, scales = "free_x") + geom_boxplot() + theme_bw()  + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")  + ylab("Classification Accuracy") + xlab("")+scale_color_manual(values=cbPalette)
print(g)
dev.off()
 










