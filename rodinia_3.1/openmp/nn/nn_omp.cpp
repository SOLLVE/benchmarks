/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <cstring>

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,unsigned long long numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
void euclid(LatLong *d_locations, float *d_distances, unsigned long long numRecords,float lat, float lng)
{
#if defined(OMP_GPU_OFFLOAD_UM)
#pragma omp target teams distribute parallel for firstprivate(lat,lng) \
  map(to: d_locations[0:numRecords]) map(from: d_distances[0:numRecords])
#elif defined(OMP_GPU_OFFLOAD)
#pragma omp target teams distribute parallel for firstprivate(lat,lng) \
  map(to: d_locations[0:numRecords]) map(from: d_distances[0:numRecords])
#else
#pragma omp parallel for firstprivate(lat,lng)
#endif
    for (unsigned long long globalId = 0; globalId < numRecords; globalId++) {
      LatLong *latLong = d_locations+globalId;
      float *dist=d_distances+globalId;
      *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
    }
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
	unsigned long long     i=0;
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;

    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;

    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device)) {
      printUsage();
      return 0;
    }

    unsigned long long numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;

    // reuse device as input_time
    if (device > 0) {
      int input_time = device;
      for (unsigned long long j = 1; j < input_time; j++) {
        for (i = 0; i < numRecords; i++) {
          locations.push_back(locations[i]);
          records.push_back(records[i]);
        }
      }
      numRecords *= input_time;
    }

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;


	/**
	* Allocate memory on host and device
	*/
#if defined(OMP_GPU_OFFLOAD_UM)
	distances = (float *)omp_target_alloc(sizeof(float) * numRecords, -100);
	d_locations = (LatLong *)omp_target_alloc(sizeof(LatLong) * numRecords, -100);
#else
	distances = (float *)malloc(sizeof(float) * numRecords);
	d_locations = (LatLong *)malloc(sizeof(LatLong) * numRecords);
#endif

    unsigned long long total_size = sizeof(LatLong) * numRecords + sizeof(float) * numRecords;
    printf("Total size: %f GB\n", (float)((total_size/1024.0)/1024.0/1024.0));

    for (i = 0; i < numRecords; i++) {
      d_locations[i] = locations[i];
    }

    double start = omp_get_wtime();

    /**
    * Execute kernel
    */
    euclid(d_locations,distances,numRecords,lat,lng);

	// find the resultsCount least distances
    findLowest(records,distances,numRecords,resultsCount);

    double end = omp_get_wtime();
	printf("Compute time: %f\n", end - start);
    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
    //Free memory
#if defined(OMP_GPU_OFFLOAD_UM)
    omp_target_free(d_locations, omp_get_default_device());
    omp_target_free(distances, omp_get_default_device());
#else
    free(distances);
	free(d_locations);
#endif
    return 0;

}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,unsigned long long numRecords,int topN){
  unsigned long long i,j;
  float val;
  unsigned long long minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
