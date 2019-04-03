corrs <- invcorrs <- structs <- NULL
corrs[[1]] <- corr.mean
invcorrs[[1]] <- prec.mean
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
	invcorrs.sp[[i]] <- invcorrs[[i]] * (structs[[i]] > structs[[i]][1,1]/2)
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

pdf("../figures/corrfit.pdf", width = 5.5, height = 5.5)
temp <- corrs[[1]]
colnames(temp) <- rep("", dim(temp)[1])
rownames(temp) <- rep("", dim(temp)[1])
corrplot(temp, method="color", 
	title = paste0("Correlation matrix"), bg = "gray", diag=T, tl.cex = 0.5, tl.srt=45, is.corr=T, mar = c(1,1,2,1)-0.9, cl.pos='r')	
dev.off()

pdf("../figures/invcorrfit.pdf", width = 5.5, height = 5.5)
temp <- invcorrs[[1]]
colnames(temp) <- rep("", dim(temp)[1])
rownames(temp) <- rep("", dim(temp)[1])
corrplot(temp, method="color", 
	title = paste0("Inverse Correlation matrix"), bg = "gray", diag=T, tl.cex = 0.5, tl.srt=45, is.corr=F, mar = c(1,1,2,1)-0.9, cl.pos='r')	
dev.off()

pdf("../figures/structfit.pdf", width = 5, height = 5)
temp <- as.matrix(structs[[1]]) / structs[[1]][1,1]
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
	freqs[i] <- structs[[1]][edgelist[i, 1], edgelist[i, 2]] / structs[[1]][1,1]
}
edges_text <- edgelist
for(i in 1:2) edges_text[, i] <- sympstext[match(edges_text[, i], symps)]

e5 <- cbind(round(freqs, 2), edges_text, round(-values, 2))
e5 <- e5[order(freqs, decreasing = T), ]
freqs.ordered <- freqs[order(freqs, decreasing = T)]
# print(xtable(e5[1:20, ]), include.rownames=FALSE)
print(xtable(e5), include.rownames=FALSE)
sum(freqs.ordered > 0.5)


############################################################
## Karonga Raw Data Summary Data Not Online
############################################################
if(FALSE){
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
}

