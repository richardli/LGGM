########################################################################
# combine the results from process_java_results_sim1A.r
########################################################################

library(xtable)
typecov <- "PX"
names <- paste0("Case", 1:4)
combine <- vector("list", length(names))
names(combine) <- names
for(pre in names){
	combine[[pre]] <- vector("list", 3)
	names(combine[[pre]]) <- c("0.0", "0.2", "0.5")
	for(miss.rate in c("0.0", "0.2", "0.5")){
		if(file.exists(paste0("rdaVA/201802/sim/", typecov, pre, miss.rate, "metrics-0309.rda"))){
			load(paste0("rdaVA/201802/sim/", typecov, pre, miss.rate, "metrics-0309.rda"))	
			combine[[pre]][[miss.rate]] <- allout
			print(paste(pre, miss.rate))
			print(apply(combine[[pre]][[miss.rate]], c(2,3), mean))			
		}
	}
}
typecov <- "SSSL"
names <- paste0("Case", 1:4)
combineSSSL <- vector("list", length(names))
names(combineSSSL) <- names
for(pre in names){
	combineSSSL[[pre]] <- vector("list", 3)
	names(combineSSSL[[pre]]) <- c("0.0", "0.2", "0.5")
	for(miss.rate in c("0.0", "0.2", "0.5")){
		if(file.exists(paste0("rdaVA/201802/sim/", typecov, pre, miss.rate, "metrics-0309.rda"))){
			load(paste0("rdaVA/201802/sim/", typecov, pre, miss.rate, "metrics-0309.rda"))	
			combineSSSL[[pre]][[miss.rate]] <- allout
			print(paste(pre, miss.rate))
			print(apply(combineSSSL[[pre]][[miss.rate]], c(2,3), mean, na.rm=T))			
		}
	}
}

# # save(combine, file = "combined-metrics.rda")

all <- NULL
for(pre in names[1:4]){
	for(miss.rate in c("0.0", "0.2", "0.5")){
        # PX
        tmp <- apply(combine[[pre]][[miss.rate]], c(2,3), mean)[1:3, c(1,2,3,9,10)]
        # SSSL
        new <- apply(combineSSSL[[pre]][[miss.rate]], c(2,3), mean, na.rm=TRUE)
        tmp <- rbind(tmp, 
        			c(new[3, c(1,2,3)], new[4, c(9,10)]))
        rownames <- rep("", 4)
     	rownames[1] <- "Semi-parametric w/o prior"
     	rownames[2] <- "Semi-parametric w/i prior"
     	rownames[3] <- "Uniform prior"
     	rownames[4] <- "Spike-and-Slab prior"
     	current <- data.frame(Scenario = pre, Missing = c(paste0(as.numeric(miss.rate) * 100, "%"), NA, NA, NA), Estimator = rownames)
     	current <- cbind(current, tmp)
     	rownames(current) <- NULL
     	all <- rbind(all, current)
     }
 }
for(i in names){
	all[which(all[,1] == i)[-1], 1] <- NA
} 

print(xtable(all[1:24, ]), include.rownames=FALSE )
print(xtable(all[25:48, ]), include.rownames=FALSE )
