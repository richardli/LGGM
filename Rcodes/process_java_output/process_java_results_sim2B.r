########################################################################
# combine the results from process_java_results_sim2A.r
########################################################################
# Count how many raw results available
name <- c("A", "B", "C", "D")
type <- c("Random")
method <- c("PX", "SSSL")[2]
N <- c(0, 80, 200, 400) + 800
allnames <- paste0(rep(name, length(type)), rep(type,each=length(name)))
Nlist <- rep(N-800, each = length(allnames))
allnames <- paste0(rep(allnames, 3), Nlist)
P <- 50
miss <- c("0.0", "0.2", "0.5")
count <- matrix(0, length(allnames), length(miss))
dir <- "../experiments/new"
for(i in 1:length(allnames)){
	for(j in 1:length(miss)){
		nn <- allnames[i]
		mm <- miss[j]			
		subdir <- paste0(nn, method, "N", N, "P", P, "Miss", mm)
		files <- list.files(paste0(dir, subdir, "/"))
		tmp <- gsub("Rep", "!", files)
		tmp <- gsub("_corr_out_mean.txt", "!", tmp)
		tmp <- unlist(strsplit(tmp, "!"))
		# force the non-rep number to be NA
		seeds <- sort(unique(as.numeric(tmp)))
		print(paste0("Total number of fit: ", length(seeds)))
		count[i, j] <- length(seeds)
	}
}
colnames(count) <- miss
rownames(count) <- allnames
count

# #  Organize results
# # srun --pty --partition=short --time=0:10:00 --mem-per-cpu=2500 /bin/bash
# # module load R
# # # R
Ntest <- 800
typecov <- c("SSSL")
name <- c("CC" ,"DD")
type <- c("Random")
miss <- c("0.0", "0.2", "0.5")
N <- c(0, 80, 200) + Ntest
counts <- array(NA, dim = c(length(name), length(miss), length(N), length(typecov), length(type)))
all <- array(NA, dim = c(length(name), length(miss), length(N), length(typecov), length(type), 6, 4))
toplot <- data.frame(case = NULL, train = NULL, miss = NULL, method = NULL, CSMF = NULL, top1 = NULL, top2 = NULL, top3 = NULL)

for(i in 1:length(name)){
for(j in 1:length(miss)){
for(k in 1:length(N)){
for(m in 1:length(typecov)){
for(n in 1:length(type)){
	ii <- name[i]
	jj <- miss[j]	
	kk <- N[k]
	mm <- typecov[m]
	nn <- type[n]
	pre <- paste0(ii, nn, kk-Ntest)	
	if(file.exists(paste0("rdaVA/201802/sim/", pre, mm, jj, "-1prediction-0309.rda"))){
			out <- array(NA, dim = c(100, 6, 4))
			start <- 1
			for(irep in 1:20){
				if(file.exists(paste0("rdaVA/201802/sim/", pre, mm, jj, "-", irep, "prediction-0309.rda"))){
					load(paste0("rdaVA/201802/sim/", pre, mm, jj, "-", irep, "prediction-0309.rda"))
					out[start : (start + 4), , ] <- metric2
				}
				start <- start + 5 					
			}
			names <- dimnames(metric2)[[2]] 
			cat(".")
	}else{
		cat("*")
		next
	}
	all[i, j, k, m, n, , ] <- apply(out, c(2, 3), mean, na.rm=TRUE)
	counts[i, j, k,m, n] <- sum(!is.na(out[,1,1]))

	pick <- c(1,2,5,6)
	case0 = rep(ii, dim(out)[1] * 4)
	train0 = rep(kk, dim(out)[1] * 4)
	miss0 = rep(jj, dim(out)[1] * 4)
	typecov0 = rep(mm, dim(out)[1] * 4)
	type0 = rep(nn, dim(out)[1] * 4)
	method0 = rep(names[pick], each = dim(out)[1])
	
	# # temporary (for v1?)
	# if(ii %in% c("I", "J") && m == 1){
	# 	out[, 4, ] <- out[, 1, ]
	# }
	
	csmf0 = as.vector(out[, pick, 1])
	top10 = as.vector(out[, pick, 2])
	top20 = as.vector(out[, pick, 3])
	top30 = as.vector(out[, pick, 4]) 
	toplot <- rbind(toplot, data.frame(case = case0, train = train0, miss = miss0, method = method0, type = type0, typecov = typecov0, csmf = csmf0, top1 = top10, top2 = top20, top3 = top30))
}
}
}
}
}
dimnames(counts)[[1]] <- dimnames(all)[[1]] <- name
dimnames(counts)[[2]] <- dimnames(all)[[2]] <- miss
dimnames(counts)[[3]] <- dimnames(all)[[3]] <- N
dimnames(counts)[[4]] <- dimnames(all)[[4]] <- typecov
dimnames(counts)[[5]] <- dimnames(all)[[5]] <- type
counts
save(toplot, file = "rdaVA/201802/sim/classification_results_random-0309.rda")
################################################################

library(ggplot2)
library(plyr)
load("rdaVA/201802/sim/classification_results_random-0309.rda")
toplot$method <- revalue(toplot$method, c("FittedS1"="Proposed0","Plug-inS1"="Proposed"))
toplot$train <- revalue(as.character(toplot$train), c("800"="GM: N.train = 0", "880"="GM: N.train = 80", "1000"="GM: N.train = 200"))
toplot$train <- factor(toplot$train, levels = c("GM: N.train = 0", "GM: N.train = 80", "GM: N.train = 200"))
toplot$caselabel <- toplot$case
toplot$caselabel <- revalue(toplot$caselabel, c("AA"="correct prior", "CC" = "correct prior", "BB" = "misspecified prior", "DD" = "misspecified prior" ))
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
palette3 <- c('#7fc97f','#beaed4',
			  '#ffffd4','#fed98e','#fe9929','#cc4c02', 
			  '#f1eef6','#d7b5d8','#df65b0','#ce1256')
palette2 <- c('#7fc97f','#beaed4',
			  '#ffffd4','#fed98e','#fe9929','#cc4c02')

toplot$miss2 <- 100 * as.numeric(as.character(toplot$miss))
toplot$miss2 <- paste0("Missing: ", toplot$miss2, "%")

toplot <- toplot[toplot$typecov == "SSSL",]
toplot$xlab <-  as.character(toplot$train)
toplot$xlab[toplot$method == "InterVA"] <- "InterVA"
toplot$xlab[toplot$method == "NaiveBayes"] <- "NaiveBayes"
order <- sort(unique(toplot$xlab))[c(4,5,1:3)]

# remove redundant
repeated <- intersect(which(toplot$method %in% c("InterVA", "NaiveBayes")), which(toplot$train != "GM: N.train = 0"))
toplot <- toplot[-repeated, ]
toplot$xlab <- factor(toplot$xlab, levels=order)


pdf("../figures/2018/Classification-mixed.pdf", width = 8, height = 6)
g <- ggplot(data = subset(toplot, 
	method %in% c( "NaiveBayes", "Proposed") & 
	case %in% c("CC", "DD") &
	type=="Random"
	), aes(x = xlab, y = top1)) 
g <- g + geom_boxplot(aes(fill = xlab), size=0.3, outlier.size=0.4)
g <- g + facet_grid(caselabel~miss2)#, scales="free")
g <- g + scale_fill_manual(values=palette3[-1], name="Method")
g <- g + xlab("Proportion of missing data")
g <- g + ylab("Accuracy")
g <- g + ggtitle("20-Group Classification Accuracy (n = 800)")  
g <- g + theme_bw()  
g <- g + theme(axis.title.x=element_blank(),
        	axis.text.x=element_blank(),
        	axis.ticks.x=element_blank(),
        	legend.position="bottom")
g <- g+guides(fill=guide_legend(nrow=1,byrow=F))
g 
dev.off()


pdf("../figures/2018/Classification-mixed-full.pdf", width = 8, height = 6)
g <- ggplot(data = subset(toplot, 
	method %in% c("InterVA", "NaiveBayes", "Proposed") & 
	case %in% c("CC", "DD") &
	type=="Random"
	), aes(x = xlab, y = top1)) 
g <- g + geom_boxplot(aes(fill = xlab), size=0.3, outlier.size=0.4)
g <- g + facet_grid(caselabel~miss2)#, scales="free")
g <- g + scale_fill_manual(values=palette3, name="Method")
g <- g + xlab("Proportion of missing data")
g <- g + ylab("Accuracy")
g <- g + ggtitle("20-Group Classification Accuracy (n = 800)")  
g <- g + theme_bw()  
g <- g + theme(axis.title.x=element_blank(),
        	axis.text.x=element_blank(),
        	axis.ticks.x=element_blank(),
        	legend.position="bottom")
g <- g+guides(fill=guide_legend(nrow=1,byrow=F))
g 
dev.off()


pdf("../figures/2018/Classification-mixed-csmf.pdf", width = 8, height = 6)
g <- ggplot(data = subset(toplot, 
	method %in% c("NaiveBayes", "Proposed") & 
	case %in% c("CC", "DD") &
	type=="Random"
	), aes(x = xlab, y = csmf)) 
g <- g + geom_boxplot(aes(fill = xlab), size=0.3, outlier.size=0.4)
g <- g + facet_grid(caselabel~miss2)
g <- g + scale_fill_manual(values=palette3[-1], name="Method")
g <- g + xlab("Proportion of missing data")
g <- g + ylab("Accuracy")
g <- g + ggtitle("20-Group CSMF Accuracy (n = 800)")  
g <- g + theme_bw()  
g <- g + theme(axis.title.x=element_blank(),
        	axis.text.x=element_blank(),
        	axis.ticks.x=element_blank(),
        	legend.position="bottom")
g <- g+guides(fill=guide_legend(nrow=1,byrow=F))
g 
dev.off()



pdf("../figures/2018/Classification-mixed-csmf-full.pdf", width = 8, height = 6)
g <- ggplot(data = subset(toplot, 
	method %in% c("InterVA","NaiveBayes", "Proposed") & 
	case %in% c("CC", "DD") &
	type=="Random"
	), aes(x = xlab, y = csmf)) 
g <- g + geom_boxplot(aes(fill = xlab), size=0.3, outlier.size=0.4)
g <- g + facet_grid(caselabel~miss2)
g <- g + scale_fill_manual(values=palette3, name="Method")
g <- g + xlab("Proportion of missing data")
g <- g + ylab("Accuracy")
g <- g + ggtitle("20-Group CSMF Accuracy (n = 800)")  
g <- g + theme_bw()  
g <- g + theme(axis.title.x=element_blank(),
        	axis.text.x=element_blank(),
        	axis.ticks.x=element_blank(),
        	legend.position="bottom")
g <- g+guides(fill=guide_legend(nrow=1,byrow=F))
g 
dev.off()

toplot2 <- melt(toplot)
toplot2$variable <- revalue(toplot2$variable, c("csmf" = "CSMF Accuracy", "top1" = "Top Class Accuracy"))
toplot2$xlab
pdf("../figures/2018/Classification-mixed-both-full.pdf", width = 10, height = 6)
g <- ggplot(data = subset(toplot2, 
	method %in% c("InterVA","NaiveBayes", "Proposed") & 
	case %in% c("CC", "DD") &
	type=="Random" &
	variable %in% c("CSMF Accuracy", "Top Class Accuracy")
	), aes(x = xlab, y = value)) 
g <- g + geom_boxplot(aes(fill = xlab), size=0.3, outlier.size=0.4)
g <- g + facet_grid(variable~miss2+caselabel)
g <- g + scale_fill_manual(values=palette3, name="Method")
g <- g + xlab("Proportion of missing data")
g <- g + ylab("Accuracy")
g <- g + ggtitle("20-Group CSMF Accuracy (n = 800)")  
g <- g + theme_bw()  
g <- g + theme(axis.title.x=element_blank(),
        	axis.text.x=element_blank(),
        	axis.ticks.x=element_blank(),
        	legend.position="bottom")
g <- g+guides(fill=guide_legend(nrow=1,byrow=F))
g 
dev.off()
