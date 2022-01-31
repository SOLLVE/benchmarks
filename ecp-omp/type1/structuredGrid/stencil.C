double* jacobi3D(int threadID, double* resultMatrix)

int main(int argc, char** argv)
{
  double coeff = 1.0/7.0;
  int startj = 0;
  int endj = 0;
  double diff;
  double global_diff =0.0;
  double communicationTime;
  double communicationTimeBegin ;
  double communicationTimeEnd;
  int i, j, k;
  int its = 0;
  double tdiff, t_start, t_end;
  double tickTime;
  int Y_size;
  double threadIdleBegin = 0.0;

  MPI_Status status;
  MPI_Status statii[4];
  MPI_Request sendLeft;
  MPI_Request sendRight;
  MPI_Request recvLeft;
  MPI_Request recvRight;
  MPI_Request requests[4];
  
  its = 0;
  
  diff = 0.0; 
  
  ///  BEGIN  AN ITERATION of JACOBI STENCIL COMPUTATION
  double previousIterationTime; 
  communicationTime = 0.0;
  if(id ==0)
    {
#pragma omp master 
      {
	t_start = MPI_Wtime();  
	previousIterationTime = t_start;
      }
    }

  while (1)
    {
      #pragma omp master
       MPI_Barrier(MPI_COMM_WORLD);
      #pragma omp barrier 
    
  if(USE_HYBRID_MPI || (numprocs > 1))
	{ 
 /*  num Processes should be greater than zero */
    #pragma omp master 
	    { 
	      /* goto BARRIER; */
	      communicationTimeBegin =  MPI_Wtime();         
	      for (i = 1; i<  X -1 ; i++)
		for (j = 1; j < Y -1; j++)
		  myRightBoundary[AA(i,j)] = u[A(i,j,Z-1)]; 
	      for (i = 1; i< X -1 ; i++)
		for (j = 1; j < Y -1; j++)
		  myLeftBoundary[AA(i,j)] = u[A(i,j ,1)]; 
	     
	      if(!BLOCKING_MPI)
		{
		  int numRequests = 0;
		  if( id > 0 ) 
		    MPI_Irecv(myLeftGhostCells, ghostSize, MPI_DOUBLE, id - 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		  if(id < p-1)
		    MPI_Irecv(myRightGhostCells, ghostSize, MPI_DOUBLE, id + 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		  if (id > 0)
		    MPI_Isend(myLeftBoundary, boundarySize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &requests[numRequests++]);		  
		  if (id < p - 1 ) 
		    MPI_Isend(myRightBoundary, boundarySize, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD, &requests[numRequests++]);

		  //	  MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
		  MPI_Waitall(numRequests, requests, statii);
		}
	      else
		{
		  if (id > 0 ) 
		      MPI_Send(myLeftBoundary, boundarySize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);  
		  if (id < p-1) 
		      MPI_Recv(myRightGhostCells, ghostSize, MPI_DOUBLE, id +1, 0, MPI_COMM_WORLD, &status ); 		 
		  if (id < p-1)
		      MPI_Send(myRightBoundary, boundarySize, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
		  if (id > 0) /* if id  is 0 we  don't  receive because the ghost cells  are actually the boundary cells */
		      MPI_Recv(myLeftGhostCells, ghostSize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
		}
	    }
	}

      tdiff = 0.0; 
      // partitioning of slabs to threads. 
      // startj = 1 + (Y/numThreads)*threadID;
      // endj = startj + Y/numThreads;

      startj = 0;
      endj = Y;
      // TODO: Add PAPI here for collecting cache misses 
    
      // TODO: figure out how to separate gpu diff from node diff .
      // TODO: figure out how to make some number of threads (one per core of multi-core each control a GPU). 
      // TODO: tune num teams , thread limit , distschedule , chunk size 
      // patition loop between CPU and GPUs 
      // map(tofrom: gpudiff) 
	  /* Note:  The variable sum is now mapped with tofrom, for correctexecution with 4.5 (and pre-4.5) compliant compilers. See Devices Intro.S-17*/ 


      // can use user-defined schedules here with target spread. 
 #pragma omp for schedule (guided)
    
   //  #pragma omp target map(alloc: u[0:(X*Y*Z)], v[0:(X*Y*Z)])  map(tofrom: gpudiff)
    // #pragma omp teams num_teams(8) thread_limit(16) 
    // #pragma omp distribute parallel for dist_schedule(static, 1024) 
    {
     for(j = startj ; j < endj ; j++)
	      for(i = 1; i < X-1; i++)
	        for (k = 1; k < Z - 1; k++) 
        		w[A(i,j,k)] = (u[A(i-1,j,k)]+ u[A(i+1,j,k)] 
			       + u[A(i,j-1,k)] + u[A(i, j+1, k)]+
			       + u[A(i, j, k-1)] + u[A(i,j, k+1)]
			       + u[A(i, j, k)] )*coeff;
      
  

#pragma omp master 
	{
	  if(POINTER_SWAP) 
	    {
	      double* temp = w ;
	      w = u; 
	      u = temp;
	    }
	  else
	    {
	      for (i = 0; i<X-1; i++)
	     	 for(j = 0; j< Y-1; j++)
		      for(k = 0; k < Z-1; k++)
		        w[A(i,j,k)] = u[A(i, j, k)];
	    }
    
      }
      // TODO : need to see if convergence check is appropriate  
	  MPI_Allreduce(&diff,  &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
	}

#pragma omp single
	its++;
	
#pragma omp barrier 

	if ( /* (global_diff <= EPSILON) */ (its >= numTimesteps) )
	  {
	    // #pragma omp master
	    #ifdef VERBOSE
	    printf("converged or completed max jacobi iterations.\n");
	    #endif 
	    break;
	  } 
	else 
	  {
#pragma omp master 
	    {
#ifdef DEBUG 
	    cout << "Process " << procID <<  " finished Jacobi iteration " << its << endl;
#endif
	    }
	  }
    } // END WHILE loop of Jacobi Iterative Computation 
  
}
