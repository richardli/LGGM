alias javaclib="javac -cp ../library/\*:../library/jdistlib-0.4.1-bin/\*"
alias javalib="java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\*"
cd ../src

javaclib math/*.java sampler/*.java util/*.java
seed=1
javalib sampler/Latent_model 200 50 0.2 10 10 SSSL Random 3000 1 $seed false true true true /Users/zehangli/Bitbucket/LGGM/experiments/ multichain 1 true 0 1 true true
