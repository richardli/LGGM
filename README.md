# LGGM
Java implementation of the methods described in 

Zehang R Li, Tyler H McCormick, and Samuel J Clark. _Bayesian Inference of Latent Gaussian Graphical Models for Mixed Data_, 2018 [arXiv](https://arxiv.org/abs/1711.00877)

## Instructions
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
name=test1 # a name of the experiment
seed=12345
java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont 1 SSSL Random 1000 1 $seed false $misspecified true $transform $dir $name 0
```

3. Process the results in R
The configurations are hard-coded in the R codes at the beginning. This script calculates the values used to generate Table 1 of the paper. (But notice the complete Table 1 is calculated as averages of a large number of replications under different cases.)
```
cd ../Rcodes
Rscript process_java_output/process_java_results_sim1A.r
```

4. Run a simple classification model described in Section 5.2
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

5. Process the results in R
The configurations are hard-coded in the R codes at the beginning. This script calculates the values used to generate Figure 2 of the paper. (But notice the complete Figure 2 is calculated as the boxplot of a large number of replications under different cases.)
```
cd ../Rcodes
Rscript process_java_output/process_java_results_sim2A.r
```

