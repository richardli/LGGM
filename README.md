# LGGM
Java implementation of the methods described in 

Zehang R Li, Tyler H McCormick, and Samuel J Clark. _Using Bayesian Latent Gaussian Graphical Models to Infer Symptom Associations in Verbal Autopsies_, 2019 [arXiv](https://arxiv.org/abs/1711.00877)

# Workflow
## Simulation: estimating the latent graphical model
1. Build Java classes using JVM at least 1.7
```
java -version
cd src
javac -cp ../library/\*:../library/jdistlib-0.4.1-bin/\* math/*.java sampler/*.java util/*.java
```

2. Run a simple model described in Section 5.1
For illustration, we only run the model 1000 times for a small problem. It takes about 1 minute on my laptop. This creates several text files of the posterior summaries in the specified directory.
```
N=200 # number of observations
P=50 # number of variables
miss=0.2 # proportion of missing data
Pcont=5 # number of continuous variables
misspecified=true # misspecification described in Sec 5.1
transform=true # misspecification of continuous variables described in Sec 5.1
dir=../data/ # direction to save the results
name=test # a name of the experiment
seed=12345
java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont 1 SSSL Random 1000 1 $seed false $misspecified true $transform $dir $name 0 false 
```

3. Process the results in R
The configurations are hard-coded in the R codes at the beginning. This script calculates the values used to generate Table 1 of the paper. (But notice the complete Table 1 is calculated as averages of a large number of replications under different cases.)
```
cd ../Rcodes
Rscript process_java_output/process_java_results_singlegroup.r
```

## Simulation: estimating the latent graphical model with classification
1. Run a simple classification model described in Section 5.2
Again we only run 1000 iterations with 5 classes. It takes about 10 minutes.
```
cd ../src
N=200 # number of observations with labels
N_test=800 # number of observations without labels
G=5 # number of classes
P=50 # number of variables
miss=0.2 # proportion of missing data
Pcont=5 # number of continuous variables
misspecified=true # misspecification described in Sec 5.1
transform=true # misspecification of continuous variables described in Sec 5.1
dir=../data/ # direction to save the results
name=test2 # a name of the experiment
seed=12345
java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_classifier $N $P $miss $Pcont 1 SSSL Random 1000 1 $seed false $misspecified true $transform $dir $name $G $N_test 0 false false true 
```

2. Process the results in R
The configurations are hard-coded in the R codes at the beginning. This script calculates the values used to generate Figure 2 of the paper. (But notice the complete Figure 2 is calculated as the boxplot of a large number of replications under different cases.)
```
cd ../Rcodes
Rscript process_java_output/process_java_results_multigroup.r
```

## PHMRC data example
1. Download and clean up the PHMRC data for the java codes to run. This creates one train-test split, and the marginal priors in the data/phmrc/ directory. For multiple train-test split, go into the exp-PHMRC.R file to change _nsample_.
```
cd ../Rcodes
Rscript data_preprocess/exp-PHMRC.R
```
2. Run a classification model on this train-test split with training data, i.e., the purple box in Figure 3. For the no training data version, change maxTrain below to 0. The following 100 iteration takes about 20 min. For real  use of the codes, more iterations are needed.
```
cd ../src
P=162 # number of symptoms
adaptive=true # adaptively estimate the variance
samepop=false # assume not sharing the same CSMF
dirichlet=true # use the Dirichlet prior
shrink=1 # do not shrink the marginal priors
anneal=false # no simulated annealing
maxTrain=10000 # max number of training data used
rep=1 # the index of replications
java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  util/ProcessVAdata phmrc/typePS$rep ../data/phmrc/PS\_0\_train$rep ../data/phmrc/PS\_0\_test$rep ../data/ 100 rep$rep SSSL 0.5 false false $P 54321 $maxTrain true $adaptive 0.01 1 10 0.0001 true $samepop $anneal nofile $shrink 1 nofile $dirichlet
```
3. Process the results in R and save relevant objects 
```
cd ../Rcodes
Rscript process_java_output/process-PHMRC.r
```