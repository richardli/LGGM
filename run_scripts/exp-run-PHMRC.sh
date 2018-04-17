javaclib math/*.java sampler/*.java util/*.java

# uninformative not same population
P=162
ii=1
adaptive=true
samepop=false
dirichlet=true
shrink=1
anneal=false
var0=1
case=20F

test=0
for type in `seq 1 50`; do
srun --mem-per-cpu=4500 --partition=medium --time=5-12:00:00 -J kcvS ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typePS$type expnew/PS\_$test\_train$type expnew/PS\_$test\_test$type ../experiments/ 3000 newPS9-$test-$type-$case SSSL 0.5 false false $P 54321 10000 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet> ../experiments/log/newP9-$test-$type-$case& 
ii=$(($ii+1))
sleep 1
done

# uninformative no training
P=162
ii=1
adaptive=true
samepop=false
dirichlet=true
shrink=1
anneal=false
var0=1
case=20E

test=0
for type in `seq 1 50`; do
srun --mem-per-cpu=4500 --partition=largemem --time=5-12:00:00 -J kcvS ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typePS$type expnew/PS\_$test\_train$type expnew/PS\_$test\_test$type ../experiments/ 3000 newPS9-$test-$type-$case SSSL 0.5 false false $P 54321 0 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet> ../experiments/log/newP9-$test-$type-$case& 
ii=$(($ii+1))
sleep 1
done

