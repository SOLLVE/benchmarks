# Repository for Performance Benchmarks for OpenMP Features 
This is a repository for benchmarks for performance experimentation of features of OpenMP 5.0 for United States Department of Energy's Exascale Computing Project's (ECP's) OpenMP project called SOLLVE. See sollve.github.io for more information on the SOLLVE project. 

The benchmarks are based on SOLLVE's efforts to undertand performance of OpenMP features, particularly OpenMP offload features. These benchmarks have been experimented on various systems as shown in the table below. 


| System                           | LLVM 11 - Summit  | LLVM 11 - Cori-GPU | LLVM 11 - Theta     | Spock           | 
|----------------------------------|:-----------------:|-----------------:|----------------------:|:--------------: |
| Stencil - 1D                     |   tested, works   |  tested, works   |   tested, works       |                 |
| Stencil - 2D                     |   tested, works   |  tested, works   |   tested, works       |                 | 
| MatMul                           |   tested, works   |  tested, works   |   tested, works       |                 | 
| Load Imbalanced MatMul           |   tested, works   |  tested          |   tested, works       |                 |
| Dot Product                      |   tested, works   |  tested          |   tested, works       |                 | 
| Stream                           |   tested          |                  |                       |                 | 
| Square Rooted Vec Prod           |   tested, works   |  tested, work    |   tested, works       |                 |
|                                  |                   |                  |                       |                 |


| System                           | LLVM 12 - Summit  | LLVM 12 - Cori-GPU | LLVM 12 - Theta     | Spock               | 
|----------------------------------|:-----------------:|-----------------:|----------------------:|:-------------------:|
| Stencil - 1D                     |   tested, works   |  tested, works   |   tested, works       |                     |
| Stencil - 2D                     |   tested, works   |  tested, works   |   tested, works       |                     |  
| MatMul                           |   tested, works   |  tested, works   |   tested, works       |                     | 
| Dot Product                      |   tested, works   |  tested          |   tested, works       |                     | 
| Stream                           |   tested          |                  |                       |                     | 
| Square Rooted Vec Prod           |   tested, works   |  tested, work    |   tested, works       |                     |
| gauss-seidel                     |   tested, works   |  tested, works   |   tested, works       |                     |
| lu factorization                 |   tested, works   |  tested, works   |  tested, works        |  tested, works      |
| spmv                             |  tested, works    |  tested, works   |  tested, works        |   tested, works     |
| quicksort                        |  tested, works    |  tested, works   |   tested, works       |   tested, works     |
| barnes-hut                       |   tested, works    |  tested, works   |   tested, works      |   tested, works     |
| Load Imbalanced MatMul           |   tested, works   |  tested          |   tested, works       |                     |

| System                           | LLVM 12 - Summit  | LLVM 12 - Cori-GPU | LLVM 12 - Theta     | Spock               | 
|----------------------------------|:-----------------:|-----------------:|----------------------:|:-------------------:|
| Stencil - 1D                     |   tested, works   |  tested, works   |   tested, works       | tested, works       |
| Stencil - 2D                     |   tested, works   |  tested, works   |   tested, works       | tested, works       | 
| MatMul                           |   tested, works   |  tested, works   |   tested, works       | tested, works       |
| Dot Product                      |   tested, works   |  tested          |   tested, works       | tested, works       | 
| Stream                           |   tested          |                  |                       |                     | 
| Square Rooted Vec Prod           |   tested, works   |  tested, works   |   tested, works       |                     |
| gauss-seidel                     |   tested, works   |  tested, works   |   tested, works       |                     |
| lu factorization                 |   tested, works   |  tested, works   |  tested, works        |  tested, works      |
| spmv                             |  tested, works    |  tested, works   |  tested, works        |   tested, works     |
| quicksort                        |  tested, works    |  tested, works   |   tested, works       |   tested, works     |
| barnes-hut                       |  tested, works    |  tested, works   |   tested, works       |   tested, works     | 
| Load Imbalanced MatMul           |   tested, works   |  tested          |   tested, works       | tested, works       |


Each benchmark uses a subset of OpenMP offload features. The stencil, matmul, dot product, square rooted vec prod use a target offload construct. The load imbalanced matrix multiplication uses a device clause with the target construct. The stream benchmark uses an experimental features of target spread for multi-GPUs along with the target construct and devices clause. 
