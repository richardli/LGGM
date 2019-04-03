javaclib math/*.java sampler/*.java util/*.java

# uninformative not same population (3/19 11:00 am) maxvalue = 5 
# 11 hr - 2400 itr, 20k -> 4 days
P=162
ii=1
adaptive=true
samepop=false
dirichlet=false
shrink=1
anneal=false
var0=1
case=20N

test=0
for type in `seq 1 50`; do
srun --mem-per-cpu=10000 --partition=medium --time=5-12:00:00 -J p1 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx16g  util/ProcessVAdata expnew/typePS$type expnew/PS\_$test\_train$type expnew/PS\_$test\_test$type ../experiments/ 20000 PHMRClong-$test-$type-$case SSSL 0.5 false false $P 54321 20000 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet false 0 10 > ../experiments/log/PHMRClong-$test-$type-$case& 
ii=$(($ii+1))
sleep 1
done

sleep 10
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
srun --mem-per-cpu=8500 --partition=medium --time=3-12:00:00 -J p0 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typePS$type expnew/PS\_$test\_train$type expnew/PS\_$test\_test$type ../experiments/ 10000 PHMRClong-$test-$type-$case SSSL 0.5 false false $P 54321 0 true $adaptive 0.01 1 10 0.0001 true $samepop $anneal nofile $shrink $var0 nofile $dirichlet true> ../experiments/log/PHMRClong-$test-$type-$case& 
ii=$(($ii+1))
sleep 1
done

