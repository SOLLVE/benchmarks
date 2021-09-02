# Repository for Performance Benchmarks for OpenMP Features 
This is a repository for benchmarks for performance experimentation of features of OpenMP 5.0 for United States Department of Energy's Exascale Computing Project's (ECP's) OpenMP project called SOLLVE. See sollve.github.io for more information on the SOLLVE project. 

The benchmarks are based on SOLLVE's efforts to undertand performance of OpenMP features, particularly OpenMP offload features. These benchmarks have been experimented on various systems as shown in the table below. 


| System                           | Summit            | Cori-GPU         | Theta                 |       | Spock | 
|----------------------------------|:-----------------:|-----------------:|----------------------:|------:|------:|
| Stencil - 1D                     |   tested, works   |  tested, works   |  tested, works        |       |       | 
| Stencil - 2D                     |   tested, works   |  tested, works   |  tested, works        |       |       | 
| MatMul                           |   tested, works   |                  |                       |       |       |
| Load Imbalanced MatMul           |   tested, works   |                  |                       |       |       | 
| Dot Product                      |   tested, works   |                  |   tested              |       |       |
| Stream                           |   tested          |                  |                       |       |       |
| Square Rooted Vec Prod           |   tested          |   tested         |                       |       |       | 


Each benchmark uses a subset of OpenMP offload features. The stencil uses, matmul and dot product use a target offload construct. The load imbalanced matrix multiplication uses a device clause with the target construct. The stream benchmark uses an experimental features of target spread for multi-GPUs along with the target construct and devices clause.
