############################################################
## Karonga 16 causes
############################################################
raw1 <- read.csv("../data/raw/alpha8_insilico_input_karonga.csv")
highcod <- raw1[, c("cod_broader1", "cod_broader2", "cod_broader3")]
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
cod0 <- as.character(highcod[, 1])
disagree = which(as.character(highcod[, 1]) != as.character(highcod[, 2])) 
cod0[disagree] <- as.character(highcod[disagree, 3])
cod0[cod0==""] <- "Cause of death unknown"
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
cod0 <- cod0[exist]

year2 <- year1
year2[year2 <= 2007] <- 0
tab <- as.matrix(table(year2, cod1))
tab0 <- table(cod1)
tab0[which(names(tab0) == "Cause of death unknown")] <- -1
order <- rev(order(tab0))
tab1 <- apply(tab, 1, function(x){x/sum(x)})
tab1 <- tab1[order,]
tab2 <- t(tab)[order, ]
causes.text <- rownames(tab2)


miss <- apply(data1, 2, function(x){sum(x == ".")})
totalmiss <- which(miss > dim(data1)[1] * 0.9)
totalmiss <- union(totalmiss, which(colnames(data) == "male"))
data1 <- data1[, -totalmiss]
data1 <- data.frame(data1)
data1 <- cbind(cod1, data1)
symps <- colnames(data1)[-1] # 94
data1[,-1] <- apply(data1[,-1], 2, function(x){x[which(x=="0")] <- "N";return(x)})
data1[,-1] <- apply(data1[,-1], 2, function(x){x[which(x=="1")] <- "Y";return(x)})
data1[,-1] <- apply(data1[,-1], 2, function(x){x[which(x=="-1")] <- ".";return(x)})
data1 <- cbind(ID = 1:dim(data1)[1], data1)

getProbbase <- function(train, G, causes){
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
 

data1[, 2] <- match(data1[, 2], causes.text)
probbase3 <- getProbbase(data1[set1, ], 16, 1:16) 
delta3 <- qnorm(1-probbase3)
data1 <- data1[, -1]
data1[,-1] <- apply(data1[,-1], 2, function(x){x[which(x=="")] <- "N";return(x)})

################################################################
# Reduce symptoms for some quick results
################################################################
allno <- apply(data1[, -1], 2, function(x){sum(x=="N")/length(x)})
allno2 <- apply(data1[year1 > 2005, -1], 2, function(x){sum(x=="N")/length(x)}) 
remain <- c(1:dim(delta)[2])[union(which(allno < 1), 
								    which(allno2 < 1))]
symps <- colnames(data1[, 1+remain]) # 92
save(symps, file = paste0("../data/expnew/sympsK.rda"))

write.table(delta3[, remain], file = paste0("../data/expnew/typeK3_delta.csv"), row.names = F, col.names = F, sep = ",", quote=F)
csmf <- as.numeric(table(data1[set1, 1]) / length(set1))
write.table(csmf, file = paste0("../data/expnew/csmf.csv"), row.names = F, col.names = F, sep = ",", quote=F)
write.table(data1[set1, c(1, 1+remain)], file = paste0("../data/expnew/K_train0.csv"), row.names = F, col.names = F, sep = ",", quote=F)
write.table(data1[-set1, c(1, 1+remain)], file = paste0("../data/expnew/K_test0.csv"), row.names = F, col.names = F, sep = ",", quote=F)
write.table(data1[, c(1, 1+remain)], file = paste0("../data/expnew/K_all0.csv"), row.names = F, col.names = F, sep = ",", quote=F)

set.seed(1234)
# Random sample of train-test
data2 <- data1[-set1, ]
order <- sample(1:dim(data2)[1], dim(data2)[1])
nn <- dim(data2)[1]
# 5%
for(i in 1:50){
	is.train <- sample(1:nn, round(nn * .05))
	is.test <- (1:nn)[-is.train]
	write.table(data2[is.train, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_train1.csv"), row.names = F, col.names = F, sep = ",", quote=F)
	write.table(data2[is.test, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_test1.csv"), row.names = F, col.names = F, sep = ",", quote=F)
}

# 10%
for(i in 1:50){
	is.train <- sample(1:nn, round(nn * .1))
	is.test <- (1:nn)[-is.train]
	write.table(data2[is.train, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_train2.csv"), row.names = F, col.names = F, sep = ",", quote=F)
	write.table(data2[is.test, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_test2.csv"), row.names = F, col.names = F, sep = ",", quote=F)
}

# 20%
for(i in 1:50){
	is.train <- sample(1:nn, round(nn * .2))
	is.test <- (1:nn)[-is.train]
	write.table(data2[is.train, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_train3.csv"), row.names = F, col.names = F, sep = ",", quote=F)
	write.table(data2[is.test, c(1, 1+remain)], file = paste0("../data/expnew/K_", i, "_test3.csv"), row.names = F, col.names = F, sep = ",", quote=F)
}

data(probbase3)
symps[symps == "skin_les"] <- "sk_les"
sympstext <- probbase3[match(symps, probbase3[, 2]), 3]
structures <- matrix(0, length(symps), length(symps))
for(i in 1:dim(probbase3)[1]){
	for(j in 4:12){
		if(probbase3[i, 2] %in% symps && probbase3[i, j] %in% symps){
			structures[match(probbase3[i, 2], symps), match(probbase3[i, j], symps)] <- 1
			structures[match(probbase3[i, j], symps), match(probbase3[i, 2], symps)] <- 1
		}
	}
}
for(i in 1:length(symps)){
	for(j in 1:length(symps)){
		if(structures[i, j] == 1){
			nei <- which(structures[, j] == 1)
			structures[i, nei] <- structures[nei, i] <- 1
		}
	}
}
colnames(structures) <- rownames(structures) <- symps
write.table(structures, file = paste0("../data/expnew/typeK_structure.csv"), row.names = F, col.names = F, sep = ",", quote=F)
 
