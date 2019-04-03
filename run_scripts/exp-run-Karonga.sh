#######################################################
##        Karonga data CV
#######################################################
P=92
samepop=false   
dirichlet=false
case=CV1
for test in `seq 1 50`; do	
# type=0 # 5% case but without training data
# srun --mem-per-cpu=8500 --partition=medium --time=3-12:00:00 -J k3 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_$test\_train1 expnew/K\_$test\_test1 ../experiments/ 20000 Karonga2019-$test-$type-$case SSSL 0.5 false false $P 54321 0 true true 0.01 1 10 0.0001 true $samepop false nofile 1 1 nofile $dirichlet false 10000 10 > ../experiments/log/KarongaCV-$type-$test-$case& 		
# 5%, 10%, 20%
for type in `seq 1 3`; do
srun --mem-per-cpu=8500 --partition=medium --time=3-12:00:00 -J k3 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_$test\_train$type expnew/K\_$test\_test$type ../experiments/ 20000 Karonga2019-$test-$type-$case SSSL 0.5 false false $P 54321 10000 true true 0.01 1 10 0.0001 true $samepop false nofile 1 1 nofile $dirichlet false 10000 10 > ../experiments/log/KarongaCV-$type-$test-$case& 	
done
sleep 1
done

##########################################################
##        Karonga data single run different starting point
###########################################################
P=92
samepop=false  
dirichlet=false  
type=0
case=1RR
srun --mem-per-cpu=8500 --partition=medium --time=2-0:00:00 -J k1 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 20000 Karonga2019-$case SSSL 0.5 false false $P 54321 10000 true true 0.01 1 10 0.0001 true $samepop false expnew/typeK_structure 1 1 nofile $dirichlet true 10000 10 > ../experiments/log/Karonga-$case&
case=2RR
srun --mem-per-cpu=8500 --partition=medium --time=2-0:00:00 -J k1 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 20000 Karonga2019-$case SSSL 0.5 false false $P 66689 10000 true true 0.01 1 10 0.0001 true $samepop false expnew/typeK_structure 1 1 nofile $dirichlet true 10000 10 > ../experiments/log/Karonga-$case&
 case=3RR
srun --mem-per-cpu=8500 --partition=medium --time=2-0:00:00 -J k1 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 20000 Karonga2019-$case SSSL 0.5 false false $P 12345 10000 true true 0.01 1 10 0.0001 true $samepop false expnew/typeK_structure 1 1 nofile $dirichlet true 10000 10 > ../experiments/log/Karonga-$case&
case=4RR
srun --mem-per-cpu=8500 --partition=medium --time=2-0:00:00 -J k1 ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx8g  util/ProcessVAdata expnew/typeK3 expnew/K\_train$type expnew/K\_test$type ../experiments/ 20000 Karonga2019-$case SSSL 0.5 false false $P 52314 10000 true true 0.01 1 10 0.0001 true $samepop false expnew/typeK_structure 1 1 nofile $dirichlet true 10000 10 > ../experiments/log/Karonga-$case&
 