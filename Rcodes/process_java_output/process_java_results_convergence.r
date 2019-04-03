library(data.table)
library(rstan)
library(RColorBrewer)
size.all <- NULL
for(rep in 1:6){
	pre <- "../experiments/multichainSSSLN200P50Miss0.2/multichainSSSLN200P50Miss0.2Rep"
	file <- paste0(pre, rep, "_mean_out.txt")
	if(file.exists(file)){
		size <- as.numeric(fread(paste0(pre, rep, "_inclusion_iteration_out.txt")))[-1]
		size.all <- rbind(size.all, size)
	}
}
Nitr <- dim(size.all)[2]
toplot <- (1:Nitr)#[(1:Nitr)%%10 == 1]
cols <- colorRampPalette(brewer.pal(9,"Set1"))(10)
pdf("../figures/conv1-size.pdf", width = 5, height = 4)
plot(toplot, size.all[1, toplot], ylim = range(size.all), type = 'l', col = alpha(cols[1], 0.6), xlab = "Iteration", ylab = "Graph size")
for(i in 2:(dim(size.all)[1])){
	lines(toplot, size.all[i, toplot], col = alpha(cols[i], 0.6))
}
dev.off()
		
counter <- 1
mean.all <- NULL
for(rep in 1:6){
	pre <- "../experiments/multichainSSSLN200P50Miss0.2/multichainSSSLN200P50Miss0.2Rep"
	file <- paste0(pre, rep, "_mean_out.txt")
	if(file.exists(file)){
		mean <- t(as.matrix(fread(paste0(pre, rep, "_mean_out.txt"))))
		mean <- cbind(ID = 1:50, mean)
		mean.all <- rbind(mean.all, mean)
	}
}

truth <- NULL
for(rep in 1:4){
	pre <- "../experiments/multichainSSSLN200P50Miss0.2/multichainSSSLN200P50Miss0.2Rep"
	file <- paste0(pre, rep, "_mean_out.txt")
	if(file.exists(file)){
		mean <- t(as.matrix(fread(paste0(pre, rep, "_mean_out_truth.txt"))))
		truth <- cbind(truth, mean)
	}
}

set.seed(1)
Nitr <- dim(mean.all)[2]-1
toplot <- (1:Nitr)#[(1:Nitr)%%10 == 1]
pdf("../figures/conv1-mean.pdf", width = 10, height = 8)
par(mfrow = c(3, 4))
plot(toplot, size.all[1, toplot], ylim = range(size.all), type = 'l', col = alpha(cols[1], 0.6), xlab = "Iteration", ylab = "Graph size", main = "Graph Size")
for(i in 2:(dim(size.all)[1])){
	lines(toplot, size.all[i, toplot], col = alpha(cols[i], 0.6))
}
for(k in sort(sample(1:50, 12-1))){
	sub <- which(mean.all[,1] == k)
	plot(toplot, mean.all[sub[1], toplot+1], ylim = range(mean.all[sub, toplot+1]), type = 'l', main = paste0("dimension ",k), col = "white", xlab = "Iteration", ylab = "latent mean")
	for(i in 1:length(sub)){
		lines(toplot, mean.all[sub[i], toplot+1], col = alpha(cols[i], 0.5))
	}
}
dev.off()
	 
sub <- which(mean.all[,1] == 1)
tt <- array(NA, dim = c(Nitr, length(sub), 50))
for(i in 1:50){
	tt[, , i] <- t(mean.all[which(mean.all[,1]==i), -1])
}		
#iterations * chains * parameters
tab <- rstan::monitor(tt, digits_summary = 3)

 