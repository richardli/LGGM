srun --pty --partition=build --time=00:02:00 --mem-per-cpu=2500 /bin/bash
# cd fortran
# make
# make install
# make clean
# cd ..
javaclib math/*.java sampler/*.java util/*.java
exit