# N=200 # number of observations
# P=50 # number of variables
# miss=0.2 # proportion of missing data
# Pcont=5 # number of continuous variables
# misspecified=true # misspecification described in Sec 5.1
# transform=true # misspecification of continuous variables described in Sec 5.1
# dir=../data/ # direction to save the results
# name=test1 # a name of the experiment
# seed=12345
# javac -cp ../library/\*:../library/jdistlib-0.4.1-bin/\* math/*.java sampler/*.java util/*.java
# java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont 1 SSSL Random 10000 1 $seed false $misspecified true $transform $dir $name 0 false 
# miss=0.5 # proportion of missing data
# name=test1b # a name of the experiment
# java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont 1 SSSL Random 10000 1 $seed false $misspecified true $transform $dir $name 0 false 


library(data.table)

pre0 <- "test0SSSLN200P50Miss0.2"
prefix <- paste0("../data/", pre0, "/", pre0, "Rep", 1)
corr.fit0 <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
mean0 <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))

pre1 <- "test1SSSLN200P50Miss0.2"
prefix <- paste0("../data/", pre1, "/", pre1, "Rep", 1)
corr.fit1 <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
mean1 <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))


pre0 <- "test0bSSSLN200P50Miss0.5"
prefix <- paste0("../data/", pre0, "/", pre0, "Rep", 1)
corr.fit0b <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
mean0b <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))

pre0 <- "test1bSSSLN200P50Miss0.5"
prefix <- paste0("../data/", pre0, "/", pre0, "Rep", 1)
corr.fit1b <- as.matrix(fread(paste0(prefix, "_corr_out_core_mean.txt"), sep = ","))
mean1b <- as.matrix(fread(paste0(prefix, "_mean_out_mean.txt"), sep = ","))





pdf("../figures/compare.pdf", width = 12, height = 12)
par(mfrow = c(2, 2))
plot(as.numeric(corr.fit0), as.numeric(corr.fit1), xlim = c(-1, 1), ylim = c(-1, 1), xlab = "Gaussian approximation", ylab = "Original likelihood", main = "Posterior mean of correlations, 20% missing", cex = .5)
abline(c(0,1))
plot(as.numeric(corr.fit0b), as.numeric(corr.fit1b), xlim = c(-1, 1), ylim = c(-1, 1), xlab = "Gaussian approximation", ylab = "Original likelihood", main = "Posterior mean of correlations, 50% missing", cex = .5)
abline(c(0,1))
r <- range(c(mean0, mean1))
plot(as.numeric(mean0), as.numeric(mean1), xlab = "Gaussian approximation", ylab = "Original likelihood", main = "Posterior mean of marginal means, 20% missing", xlim = r, ylim = r)
abline(c(0,1))
r <- range(c(mean0b, mean1b))
plot(as.numeric(mean0b), as.numeric(mean1b), xlab = "Gaussian approximation", ylab = "Original likelihood", main = "Posterior mean of marginal means, 50% missing", xlim = r, ylim = r)
abline(c(0,1))
dev.off()