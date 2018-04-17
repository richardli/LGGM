library(ggplot2)
library(scales)

med <- mean <- lower <- upper <- NULL
dim <- v0s <- v1s <- lambdas <- ps <- NULL
# for(P in c(100)){
# for(v0 in c(0.001, 0.005, 0.001, 0.05, 0.1)){
# for(h in c(10, 20, 50, 100, 150, 200)){
# for(lambda in c("2.0", "5.0", "10.0", "20.0", "30.0")){
# for(p in c("0.01", "0.001", "1.0E-4", "5.0E-5", "1.0E-5")){
# 	v1 <- v0 * h
# 	v1 <- paste0(v1, ".0")
files <- list.files("~/Sparsity/sendback/")
message <- round(seq(1, length(files), len = 11))[-1]
for(file in files){
	tmp <- gsub("Sparsity", "", file)
	tmp <- gsub("_sparsity.txt", "", tmp)
	tmp <- gsub("E-", "xx", tmp)
	tmp <- strsplit(tmp, "-")[[1]]
	for(i in 1:length(tmp)) tmp[i] <- gsub("xx", "E-", tmp[i])

	if(as.numeric(tmp[5]) < 100) next
	spar <- as.numeric(read.csv(paste0("~/Sparsity/sendback/", file), header=F))
	Nitr <- length(spar)
	med <- c(med, median(spar[(Nitr/2+1):Nitr]))
	mean <- c(mean, mean(spar[(Nitr/2+1):Nitr]))
	lower <- c(lower, quantile(spar[(Nitr/2+1):Nitr], 0.05))
	upper <- c(upper, quantile(spar[(Nitr/2+1):Nitr], 0.95))
	dim <- c(dim, as.numeric(tmp[5]))
	v0s <- c(v0s, as.numeric(tmp[1]))
	v1s <- c(v1s, as.numeric(tmp[2]))
	lambdas <- c(lambdas, as.numeric(tmp[3]))
	ps <- c(ps, as.numeric(tmp[4]))
	cat(".")
	if(length(ps) %in% message){
		print(length(ps))
	}
}
out <- data.frame(cbind(dim, v0s, v1s, lambdas, ps, mean, med, lower, upper))
out$h <- out$v1 / out$v0
out$Lambda <- factor(out$lambdas, levels = sort(unique(out$lambdas)))
out$hh <- factor(out$h, levels = sort(unique(out$h)))
out$v1 <- factor(out$v1s, levels = sort(unique(out$v1s)))
out$v1 <- paste("v1 =", out$v1)

pdf("../figures/2018/chooseLambda.pdf", width = 10, height = 10)
md <- position_dodge(width = 0.1)
g <- ggplot(data = subset(out, v0s %in% c(0.001, 0.005, 0.01, 0.1) & dim==100 & lambdas %in% c(5, 10, 20, 30) & hh %in% c(10, 50, 100, 200)), 
	aes(x = ps, y = med, color=Lambda, shape=Lambda))
g <- g + geom_point(position = md)
g <- g + geom_line(position = md) 
g <- g + scale_x_log10(expression(pi[delta]),
        breaks = c(1e-3, 1e-2, 1e-4, 1e-5),
        labels = trans_format("log10", math_format(10^.x)))
g <- g + facet_grid( v0s ~ hh )
g <- g + geom_errorbar(aes( ymin = lower, ymax = upper),position = md, size = 0.4, width = 0.1, alpha =0.5) 
g <- g + scale_color_discrete(expression(lambda))
g <- g + scale_shape_discrete(expression(lambda))
g <- g + ylab("Implied median edge probability")
# g <- g + ggtitle(expression(paste("Implied prior edge probability, ", v[0], " = 0.01")))
g <- g + theme_bw() #+ theme(legend.position = "bottom")
g
dev.off()

pdf("../figures/2018/chooseV.pdf", width = 10, height = 4)
out$v0 <- paste("v0 =", out$v0s)
md <- position_dodge(width = .2)
g <- ggplot(data = subset(out,   dim==100 & lambdas == 10 & v0s %in% c(0.001, 0.005, 0.01, 0.05, 0.1) & h %in% c(10, 50, 100, 200)), 
	aes(x = ps, y = med, color=hh, shape=hh))
g <- g + geom_point(position = md)
g <- g + geom_line(position = md) 
g <- g + geom_errorbar(aes( ymin = lower, ymax = upper),position = md, size = 0.4, width = 0.1, alpha =0.5) 
g <- g + scale_x_log10(expression(pi[delta]),
        breaks = c(1e-3, 1e-2, 1e-4, 1e-5),
        labels = trans_format("log10", math_format(10^.x)))
# g <- g + geom_hline(yintercept=0.02, color="red", linetype = "dashed")
g <- g + facet_wrap( ~ v0, nrow = 1)
g <- g + scale_color_discrete(expression(v[1]/v[0]))
g <- g + scale_shape_discrete(expression(v[1]/v[0]))
g <- g + ylab("Edge probability")
g <- g + ggtitle(expression(paste("Implied prior edge probability, ", lambda, " = 10")))
g <- g + theme_bw()
g
dev.off()


######################################################
## Complete graph
######################################################
if(FALSE){
library(ggplot2)
library(data.table)
dat <- NULL
for(v in c("0.5", "2.0")){
	for(lambda in c("1.0", "5.0", "10.0")){
		if(!file.exists(paste0("~/Comp/Compv1", v, "lambda", lambda, "_prec_out.txt"))){
			cat(v," ",lambda,"\n")
			next
		}

		invcor0 <- as.matrix(fread(paste0("~/Comp/Compv1", v, "lambda", lambda, "_prec_out.txt"), sep = ","))
		corr0 <- as.matrix(fread(paste0("~/Comp/Compv1", v, "lambda", lambda, "_corr_out.txt"), sep = ","))
		structs <- read.csv(paste0("~/Comp/Compv1", v, "lambda", lambda, "_inclusion_out.txt"), header=F)


		invcor0 <- invcor0[-(1:dim(invcor0)[2]), ]
		corr0 <- corr0[-(1:dim(corr0)[2]), ]
		invcor <- array(NA, c(dim(invcor0)[1]/dim(invcor0)[2], dim(invcor0)[2], dim(invcor0)[2]))
		corr <- invcor
		P <- dim(invcor)[2]
		counter <- 0
		for(i in 1:dim(invcor)[1]){
			invcor[i, , ] <- invcor0[(counter+1) : (counter + P), ]
			corr[i, , ] <- corr0[(counter+1) : (counter + P), ]
			# invcor[i, , ] <- solve(corr[i, , ])
			counter <- counter + P
		}
		alledges <- nonedges <- diags <- offR <- NULL
		for(i in 1:(dim(invcor)[2]-1)){
			for(j in (i+1):dim(invcor)[2]){
				if(structs[i,j] > 0){
					alledges <- c(alledges, invcor[, i, j])
				}else{
					nonedges <- c(nonedges, invcor[, i, j])
				}
				offR <- c(offR, corr[, i, j])
			}
			diags <- c(diags, invcor[, i, i])
		}
		diags <- c(diags, invcor[, P, P])

		dat0 <- data.frame(
			dens = c(diags, nonedges, alledges, offR),
			type = c(rep("Diagonal", length(diags)), 
			 rep("Non-edges", length(nonedges)), 
			 rep("Edges", length(alledges)), 
			 rep("Correlation", length(offR))))
		dat0$v <- as.numeric(v)
		dat0$lambda <- as.numeric(lambda)
		dat <- rbind(dat, dat0)
	}
}
dat$parameter <- paste0("v=", dat$v, ", lambda=", dat$lambda)
save(dat, file = "~/Comp/combined.rda")
}
load("~/Comp/combined.rda")
dat <- dat[dat$parameter != "v=0.5, lambda=1", ]
dat$parameter <- factor(dat$parameter, levels=
	c("v=0.5, lambda=5",  
	  "v=0.5, lambda=10",
	  "v=2, lambda=5",
	  "v=2, lambda=10"))
labels <- c( 
	  expression(paste("v=0.5, ",lambda,"=5")),  
	  expression(paste("v=0.5, ",lambda,"=10")),
	  expression(paste("v=2, ",lambda,"=5")),
	  expression(paste("v=2, ",lambda,"=10")))
pdf("../figures/2018/comp1.pdf", width = 6, height=4)
g1 <- ggplot(data = subset(dat,type=="Diagonal")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.3))+ggtitle(expression(paste("Marginal prior for diagonal elements in ", R^{-1}))) + xlab(expression(paste(r^{jj}))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Diagonal"], c(0, 0.975)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()
 
pdf("../figures/2018/comp2.pdf", width = 6, height=4) 
g1 <- ggplot(data = subset(dat,type== "Edges")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.3))+ggtitle(expression(paste("Marginal prior for off-diagonal elements in ", R^{-1}))) + xlab(expression(paste(r^{jk}))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Edges"], c(0.005, 0.995)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()


pdf("../figures/2018/comp3.pdf", width = 6, height=4) 
g1 <- ggplot(data = subset(dat,type== "Correlation")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.7))+ggtitle(expression(paste("Marginal prior for off-diagonal elements in ", R))) + xlab(expression(paste(R[jk]))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Correlation"], c(0.005, 0.995)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()

######################################################
## AR1
######################################################
if(FALSE){
dat <- NULL
for(v in c("0.001", "0.01", "0.1")){
	for(lambda in c("1.0", "5.0", "10.0")){
		if(!file.exists(paste0("~/AR1/AR1v0", v, "lambda", lambda, "_prec_out.txt"))){
			cat(v," ",lambda,"\n")
			next
		}

		invcor0 <- as.matrix(fread(paste0("~/AR1/AR1v0", v, "lambda", lambda, "_prec_out.txt"), sep = ","))
		corr0 <- as.matrix(fread(paste0("~/AR1/AR1v0", v, "lambda", lambda, "_corr_out.txt"), sep = ","))
		structs <- read.csv(paste0("~/AR1/AR1v0", v, "lambda", lambda, "_inclusion_out.txt"), header=F)


		invcor0 <- invcor0[-(1:dim(invcor0)[2]), ]
		corr0 <- corr0[-(1:dim(corr0)[2]), ]
		invcor <- array(NA, c(dim(invcor0)[1]/dim(invcor0)[2], dim(invcor0)[2], dim(invcor0)[2]))
		corr <- invcor
		P <- dim(invcor)[2]
		counter <- 0
		for(i in 1:dim(invcor)[1]){
			invcor[i, , ] <- invcor0[(counter+1) : (counter + P), ]
			corr[i, , ] <- corr0[(counter+1) : (counter + P), ]
			# invcor[i, , ] <- solve(corr[i, , ])
			counter <- counter + P
		}
		alledges <- nonedges <- diags <- offR <- NULL
		for(i in 1:(dim(invcor)[2]-1)){
			for(j in (i+1):dim(invcor)[2]){
				if(structs[i,j] > 0){
					alledges <- c(alledges, invcor[, i, j])
				}else{
					nonedges <- c(nonedges, invcor[, i, j])
				}
				offR <- c(offR, corr[, i, j])
			}
			diags <- c(diags, invcor[, i, i])
		}
		diags <- c(diags, invcor[, P, P])

		dat0 <- data.frame(
			dens = c(diags, nonedges, alledges, offR),
			type = c(rep("Diagonal", length(diags)), 
			 rep("Non-edges", length(nonedges)), 
			 rep("Edges", length(alledges)), 
			 rep("Correlation", length(offR))))
		dat0$v <- as.numeric(v)
		dat0$lambda <- as.numeric(lambda)
		dat <- rbind(dat, dat0)
	}
}

dat$parameter <- paste0("v0=", dat$v, ", lambda=", dat$lambda)
save(dat, file = "~/AR1/combined.rda")
}
load("~/AR1/combined.rda")
dat$parameter <- factor(dat$parameter, levels=
	c(#"v0=0.001, lambda=1",  
	  "v0=0.01, lambda=1",
	  "v0=0.1, lambda=1",
	  #"v0=0.001, lambda=10",  
	  "v0=0.01, lambda=10",
	  "v0=0.1, lambda=10"))
labels <- c( 
	  #expression(paste(v[0], "=", 10^{-3},", ", lambda,"=1")),
	  expression(paste(v[0], "=", 10^{-2},", ", lambda,"=1")),
	  expression(paste(v[0], "=", 10^{-1},", ", lambda,"=1")),
	 # expression(paste(v[0], "=", 10^{-3},", ", lambda,"=10")),
	  expression(paste(v[0], "=", 10^{-2},", ", lambda,"=10")),
	  expression(paste(v[0], "=", 10^{-1},", ", lambda,"=10")))

dat <- dat[!is.na(dat$parameter), ]

pdf("../figures/2018/sp1.pdf", width = 6, height=4) 
g1 <- ggplot(data = subset(dat,type=="Diagonal")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.5))+ggtitle(expression(paste("Marginal prior for diagonal elements in ", R^{-1}))) + xlab(expression(paste(r^{jj}))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Diagonal"], c(0, 0.975)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()

pdf("../figures/2018/sp2.pdf", width = 6, height=4) 
g1 <- ggplot(data = subset(dat,type== "Edges")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.3))+ggtitle(expression(paste("Marginal prior for non-zero diagonal elements in ", R^{-1}))) + xlab(expression(paste(r^{jk}))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Edges"], c(0.005, 0.995)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()

pdf("../figures/2018/sp3.pdf", width = 6, height=4) 
g1 <- ggplot(data = subset(dat,type== "Non-edges")) + geom_density(aes(x = dens, color = parameter, fill=parameter), alpha = .2)+ geom_hline(yintercept=0, color="white", size=0.8) +theme_classic() + theme(legend.position = c(0.8, 0.3))+ggtitle(expression(paste("Marginal prior for zero diagonal elements in ", R^{-1}))) + xlab(expression(paste(r^{jk}))) 
g1 <- g1 + xlim(quantile(dat$dens[dat$type=="Non-edges"], c(0.005, 0.995)))
g1 <- g1 + scale_color_discrete(labels = labels)
g1 <- g1 + scale_fill_discrete(labels = labels)
g1 <- g1 + theme(legend.text.align = 0)
g1
dev.off()


 

 