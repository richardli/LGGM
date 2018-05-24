# README
This folder contains files for implementation on a slurm cluster. Unfortunately since a single run of the model takes a long time, running them all on a single computer is inefficient. 

All the scripts are run in the /src directory with the following alias defined in .bashrc for easier interpretation
```
alias javaclib="javac -cp ../library/\*:../library/jdistlib-0.4.1-bin/\*"
alias javalib="java -cp .:../library/\*:../library/jdistlib-0.4.1-bin/\*"
```
