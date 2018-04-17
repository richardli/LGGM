# This scripts contains the simulation setup for latent GGM no classification first
#  Summary: 
#    Case1. correct prior, all binary 
#    Case3. correct prior, mixed
#    Case2. wrong prior,  all binary
#    Case4. wrong prior (both prob and transformation), mixed 
# srun --pty --partition=short --time=12:00:00 --mem-per-cpu=2500 /bin/bash
# javaclib math/*.java sampler/*.java util/*.java


dir=../experiments/
N=200
P=50
seed=1234
Nrep=20
for rep in `seq 1 5`; do
	Pcont=0
	misspecified=false
	transform=false
	name=Case1
	rep0=$(($rep * 1000))
	for miss in 0.0 0.2 0.5; do
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J S1-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep SSSL Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/s-$name-$miss-$twoend-$transform-$rep&
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J X1-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep PX Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/x-$name-$miss-$twoend-$transform-$rep&
	sleep 1
	done
done


sleep 10


for rep in `seq 1 5`; do
	Pcont=0
	misspecified=true
	transform=false
	name=Case2
	rep0=$(($rep * 1000))
	for miss in 0.0 0.2 0.5; do
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J S2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep SSSL Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/s-$name-$miss-$twoend-$transform-$rep&
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J X2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep PX Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/x-$name-$miss-$twoend-$transform-$rep&		
	sleep 1
	done
done

sleep 10



for rep in `seq 1 6`; do
	Pcont=5
	misspecified=false
	transform=false
	name=Case3
	rep0=$(($rep * 1000))
	for miss in 0.0 0.2 0.5; do
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J S2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep SSSL Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/s-$name-$miss-$twoend-$transform-$rep&
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J X2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep PX Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/x-$name-$miss-$twoend-$transform-$rep&		
	sleep 1
	done
done


sleep 10

for rep in `seq 1 6`; do
	Pcont=5
	misspecified=true
	transform=true
	name=Case4
	rep0=$(($rep * 1000))
	for miss in 0.0 0.2 0.5; do
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J S2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep SSSL Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/s-$name-$miss-$twoend-$transform-$rep&
		srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J X2-$miss  ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g sampler/Latent_model $N $P $miss $Pcont $Nrep PX Random 3000 1 $seed false $misspecified true $transform $dir $name $rep0 > ../experiments/log/x-$name-$miss-$twoend-$transform-$rep&		
	sleep 1
	done
done
sleep 10

sleep 120

# This scripts contains the simulation setup where
#   1. training data 0, 200, 400
#   2. 800 testing data, 50 symptoms
#   3. misspecified marginal priors
#   4. sparse and dense precision matrix
#   5. case 1, 2, 3, 4, but all misspecified not two end
#   6. add case 5(E) where continous variables are added but not transformed
#   7. redo case 2(F) and 3(G), with true condprob not in 15 levels only 
# 
#  Summary: 
#    AA. correct prior, all binary 
#    BB. correct prior, mixed
#    CC. wrong prior,  all binary
#    DD. wrong prior, mixed 
# 
#
# 
# srun --pty --partition=short --time=12:00:00 --mem-per-cpu=2500 /bin/bash
# javaclib math/*.java sampler/*.java util/*.java

dir=../experiments/
N=0
P=50
Nrep=5
G=20
N_test=800
seed=1

Pcont=0
misspecified=false
transform=false
name=A
for i in `seq 1 10`; do
	for type in Random; do
	for N in 0 80 200; do
 	for miss in 0.0 0.2 0.5; do
        seed=$(($i * 1000))
    	rep0=$(($i * 200))
    	srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J SSSL$name$miss$type ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g  sampler/Latent_classifier $N $P $miss $Pcont $Nrep SSSL $type 3000 1 $seed false $misspecified true $transform $dir new$name$type$N $G $N_test $rep0 false false true > ../experiments/log/+$name-$type-$miss-$misspecified-$transform-$seed-S$N&
    	sleep 2
	done	
	done
	done
done
echo ---------------- Finish First Part Sparse Jan 21 setup ----------------
sleep 60


dir=../experiments/
N=0
P=50
Nrep=5
G=20
N_test=800
seed=1
Pcont=0
misspecified=true
transform=false
name=B

for i in `seq 1 10`; do
	for type in Random; do
	for N in 0 80 200 400; do
 	for miss in 0.0 0.2 0.5; do
        seed=$(($i * 1000))
    	rep0=$(($i * 200))
    	srun --mem-per-cpu=2500 --partition=short --time=12:00:00 -J SL$name$miss$type ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx2g  sampler/Latent_classifier $N $P $miss $Pcont $Nrep SSSL $type 3000 0.5 $seed false $misspecified true $transform $dir new$name$type$N $G $N_test $rep0 false false true > ../experiments/log/+$name-$type-$miss-$misspecified-$transform-$seed-S$N&

    	sleep 2
	done	
	done
	done
done
echo ---------------- Finish Second Part Sparse Jan 21 setup ----------------
sleep 60

dir=../experiments/
P=50
Nrep=5
G=20
N_test=800
seed=1

Pcont=5
misspecified=false
transform=false
name=CC
for i in `seq 1 10`; do
	for type in Random; do
	for N in 0 80 200; do
 	for miss in 0.0 0.2 0.5; do
        seed=$(($i * 1000))
    	rep0=$(($i * 200))
    	srun --mem-per-cpu=4500 --partition=short --time=12:00:00 -J $name$miss ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  sampler/Latent_classifier $N $P $miss $Pcont $Nrep SSSL $type 3000 0.5 $seed false $misspecified true $transform $dir new$name$type$N $G $N_test $rep0 false false true false > ../experiments/log/+$name-$type-$miss-$misspecified-$transform-$seed-S$N&

	done	
    sleep 1
	done
	done
done

dir=../experiments/
P=50
Nrep=5
G=20
N_test=800
seed=1

Pcont=5
misspecified=true
transform=true
name=DD
for i in `seq 1 10`; do
	for type in Random; do
	for N in 0 80 200; do
 	for miss in 0.0 0.2 0.5; do
        seed=$(($i * 1000))
    	rep0=$(($i * 200))
    	srun --mem-per-cpu=4500 --partition=short --time=12:00:00 -J $name$miss ~/jdk1.8.0_111/bin/java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\* -Xmx4g  sampler/Latent_classifier $N $P $miss $Pcont $Nrep SSSL $type 3000 0.5 $seed false $misspecified true $transform $dir new$name$type$N $G $N_test $rep0 false false true > ../experiments/log/+$name-$type-$miss-$misspecified-$transform-$seed-S$N&

	done	    	
	sleep 1
	done
	done
done
