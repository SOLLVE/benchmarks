# Repo for the ECP OpenMP Monte Carlo Benchmark 

## Overview 
This is a part of the repository of the ECP OpenMP performance benchmark suite, used for a benchmark-style (smaller than mini-App style) performance experimentation based on applications with the most time consuming computation kernels having the Computational Pattern of Monte Carlo Methods.

The benchmark is primarily representative of Applications of important to the United States Department of Energy's Exascale Computing Project(ECP), in particular Autodock.

The performance experimentation we are focused on is of OpenMP 5.0 features and performance optimization based on them, and is driven the ECP's Software Technology project on developing LLVM's OpenMP, otherwise known as SOLLVE (https://www.bnl.gov/compsci/projects/SOLLVE/).

## Description of Code

The code bench.c is the actively developed version. The code bench_works.c is the stable version. That's version that should be considered by new developers working on this repository or viewers of this repository. 


## Short Link
For easy access, the short link to this page is https://tinyurl.com/mc-omp-bench .


## Development History
We note that this was originally developed at: https://bitbucket.org/tony-curtis/oscar-bench/src/master/. It was developed there originally as it was easiest for us to keep it closed source there, but now we want to make it open-source and github is the best for that for us. 
