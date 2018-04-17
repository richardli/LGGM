# Build input files from PHMRC
# (cause 11, 14, 15, 16, 24, 27, 30, 33 are external)
# install_github("richardli/openVA/openVA")
library(openVA)
PHMRC_all <- read.csv(getPHMRC_url("adult"))
# PHMRC_all2 <- read.csv(getPHMRC_url("child"))

causes <-  as.character(PHMRC_all[(match(1:34, PHMRC_all$va34)), "gs_text34"])
causes <- causes[-c(11, 14, 15, 16, 24, 27, 30, 33)]

PHMRC_all_rm <- PHMRC_all[PHMRC_all$gs_text34 %in% causes, ]
PHMRC_all_rm$gs_text34 <- as.character(PHMRC_all_rm$gs_text34)
PHMRC_all_rm$va34 <- match(PHMRC_all_rm$gs_text34, causes)

sites <- as.character(unique(PHMRC_all$site))


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

########################################################
# Experiment with simple random sampling
########################################################
set.seed(1)
nn <- 1000
for(i in 1:50){
	# adult
	is.test <- sample(1:dim(PHMRC_all_rm)[1], nn*2)
	
	test <- PHMRC_clean[is.test, ]
	train <- PHMRC_clean[-is.test, ]
	
	probbase <- getProbbase3(train, 26, 1:26)
	delta <- qnorm(1-probbase)
	write.table(delta, file = paste0("../data/expnew/typePS", i, "_delta.csv"), row.names = F, col.names = F, sep = ",", quote=F)

	train <- train[, -1]
	test <- test[, -1]
	train[,-1] <- apply(train[,-1], 2, function(x){x[which(x=="")] <- "N";return(x)})
	test[,-1] <- apply(test[,-1], 2, function(x){x[which(x=="")] <- "N";return(x)})

	test1 <- test[1:nn, ]
	test2 <- test[(nn+1):(nn+nn), ]
	write.table(test1, file = paste0("../data/expnew/PS_", 0, "_train", i, ".csv"), row.names = F, col.names = F, sep = ",", quote=F)
	write.table(test2, file = paste0("../data/expnew/PS_", 0, "_test", i, ".csv"), row.names = F, col.names = F, sep = ",", quote=F)		
	csmf <- as.numeric(table(train[, 1]) / dim(train)[1])
	write.table(csmf, file = paste0("../data/expnew/PS", i, "csmf.csv"), row.names = F, col.names = F, sep = ",", quote=F)
}

########################################################
# Experiment with simple random sampling
########################################################
set.seed(1)
nn <- 1000
for(i in 1:50){
	# adult
	is.test <- sample(1:dim(PHMRC_all_rm)[1], nn*2)
	
	test <- PHMRC_clean[is.test, ]
	train <- PHMRC_clean[-is.test, ]
	
	probbase <- getProbbase(train, 26, 1:26)
	delta <- qnorm(1-probbase)
	write.table(delta, file = paste0("../data/expnew/typePR", i, "_delta.csv"), row.names = F, col.names = F, sep = ",", quote=F)

	train <- train[, -1]
	test <- test[, -1]
	train[,-1] <- apply(train[,-1], 2, function(x){x[which(x=="")] <- "N";return(x)})
	test[,-1] <- apply(test[,-1], 2, function(x){x[which(x=="")] <- "N";return(x)})

	test1 <- test[1:nn, ]
	test2 <- test[(nn+1):(nn+nn), ]
	write.table(test1, file = paste0("../data/expnew/PR_", 0, "_train", i, ".csv"), row.names = F, col.names = F, sep = ",", quote=F)
	write.table(test2, file = paste0("../data/expnew/PR_", 0, "_test", i, ".csv"), row.names = F, col.names = F, sep = ",", quote=F)		
	csmf <- as.numeric(table(train[, 1]) / dim(train)[1])
	write.table(csmf, file = paste0("../data/expnew/PR", i, "csmf.csv"), row.names = F, col.names = F, sep = ",", quote=F)
}
  
