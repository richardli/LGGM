#######################################################
#######################################################
##        Prior simulation
#######################################################
#######################################################
javaclib math/*.java sampler/*.java util/*.java

ii=1
for P in 100 150; do
for v0 in 0.001 0.005 0.01 0.05 0.1; do #  0.01 0.02 0.03 0.05 0.1
for h in 10 20 50 100 150 200; do
for lambda in 1 5 10 20 30; do
srun --mem-per-cpu=4500 --partition=short --time=10:00:00 -J pc$ii ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  sampler/Latent_model_prior ../experiments/ $P $v0 $h $lambda 2000 > ../experiments/log/prior-$ii &
ii=$(($ii+1))
sleep 1
done
done
done
done

P=100
for v0 in 0.001 0.005; do
lambda=10
for h in 10 50 100 200; do
srun --mem-per-cpu=4500 --partition=short --time=3:00:00 -J pcX ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  sampler/Latent_model_prior ../experiments/ $P $v0 $h $lambda 2000 > ../experiments/log/prior-$ii &
done
done
 
for v0 in 0.01 0.05 0.1; do
h=200
srun --mem-per-cpu=4500 --partition=short --time=3:00:00 -J pcX ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  sampler/Latent_model_prior ../experiments/ $P $v0 $h $lambda 2000 > ../experiments/log/prior-$ii &
done

#######################################################
#######################################################
##        Karonga data
#######################################################
#######################################################
### uninformative prior
P=92
ii=1
adaptive=true
samepop=true   
dirichlet=true
shrink=1
anneal=false
var0=1

case=C 
for test in `seq 1 20`; do
type=0 # 5% case but without training data
srun --mem-per-cpu=4500 --partition=short --time=12:00:00 -J K2 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_$test\_train1 expnew/K\_$test\_test1 ../experiments/ 3000 newK9-$test-$type-$case SSSL 0.5 false false $P 54321 0 true $adaptive 0.05 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet> ../experiments/log/newK9-$test-$type-$case& 	
for type in `seq 1 3`; do
srun --mem-per-cpu=4500 --partition=short --time=12:00:00 -J K2 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_$test\_train$type expnew/K\_$test\_test$type ../experiments/ 3000 newK9-$test-$type-$case SSSL 0.5 false false $P 54321 10000 true $adaptive 0.05 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet> ../experiments/log/newK9-$test-$type-$case& 
done
sleep 1
done

# single run, first half training data, all second half testing data 
P=92
ii=1
adaptive=true
samepop=false  
dirichlet=true
shrink=1
anneal=false
var0=1
test=20
type=0
case=20F
srun --mem-per-cpu=4500 --partition=short --time=8:00:00 -J kcv0 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 3000 newK9-0-$type-$case SSSL 0.5 false false $P 54321 10000 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal expnew/typeK_structure $shrink $var0 nofile $dirichlet> ../experiments/log/newK9-$test-$type-$case&
case=21F # no training data
srun --mem-per-cpu=4500 --partition=short --time=8:00:00 -J kcv0 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 3000 newK9-0-$type-$case SSSL 0.5 false false $P 54321 0 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal expnew/typeK_structure $shrink $var0 nofile $dirichlet> ../experiments/log/newK9-$test-$type-$case&
