#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <malloc.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <fstream>

using namespace std;


#define NX 400 	                        //Number of grid points along the length
#define NY 400	                        //Number of grid points along the height
#define NZ 400
#define NUMSPD 27		                //D directions===Q_LBM
#define RHO0 1.0		                // initial density
//#define NTHREAD 1024	                //No. of threads
//#define NBLOCK 256	                //No. of blocks
#define TAU 0.6		                    //	Re=6000
#define U0 0.05		                    // velocity
#define PI acos(-1)		                // velocity
#define OUTPUTFILENAME "z_line_res.dat"
#define D_LBM 3                         // 3D
#define NNODES NX*NY*NZ                 // Total number of grid points
#define TPB2D 16


//****** Structure of Array (SOA) ******

struct params
{
	float f[NNODES * NUMSPD];                           //collision distribution functions
	float ux;
	float uy;
	float uz;
	float rho;
};
params* domain;

__constant__ float  dev_w[NUMSPD];                  // weight functions on device

__constant__ int dev_ex[NUMSPD];                    // lattice velocity on device
__constant__ int dev_ey[NUMSPD];
__constant__ int dev_ez[NUMSPD];

__constant__ int dev_kb[NUMSPD];

int Ord3(int x, int y, int z, int nx, int ny, int nz);
void Initialize(float* f);
void Outp(const float* rho, const float* ux, const float* uy, const float* uz, const float* f);
void OutWatch(int t, float uxt, float rhot);

//***** Functions called from device and executed on device *****

__device__ int d_Ord3(int x, int y, int z, int nx, int ny, int nz);
__device__ void d_Ord3r(int id, int* x, int* y, int* z, int ny, int nx);

//***** Functions called from host and executed on device *****

__global__ void Watch(float* rho_d, float* ux_d, float* uxt, float* rhot);

__global__ void collideLBGK(float* f,float* ux_d, float* rho_d);

__global__ void stream_Xforward(float* f,const int i);
__global__ void stream_Xbackward(float* f,const int i);
__global__ void stream_Yforward(float* f,const int j);
__global__ void stream_Ybackward(float* f,const int j);
__global__ void stream_Zforward(float* f,const int k);
__global__ void stream_Zbackward(float* f,const int k);


//***** Weight Functions (host) ******

const float w[NUMSPD] = { 8.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, \
1.0 / 54.0 , 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,\
1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 216.0 , 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,\
1.0 / 216.0, 1.0 / 216.0 };

//***** Lattice velocity (host) ******
int ex[NUMSPD]={0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0,1,1,1,1,-1,-1,-1,-1};
int ey[NUMSPD]={0,0,0,1,-1,0,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1};
int ez[NUMSPD]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};

int kb[NUMSPD]={0,2,1,4,3,6,5,10,9,8,7,14,13,12,11,18,17,16,15,26,25,24,23,22,21,20,19};

int main()
{
	clock_t start1, finish1,start2,finish2;

//****** Memory allocation on host *******

	long long int num_cells = pow(NX + 2, D_LBM);                             // number of grid points
	long long int field_size = NUMSPD * num_cells * sizeof(float);            // memory size of distributions
	float* f = (float*)malloc(field_size);                          // post coll. distribution
 
	float* ux = (float*)malloc(sizeof(float));
	float* uy = (float*)malloc(sizeof(float));
	float* uz = (float*)malloc(sizeof(float));
	float* rho = (float*)malloc(sizeof(float));



    Initialize(f);  // Initializing distributions on host



//****** Memory allocation for device ******

	float* f_d = (float*)malloc(field_size);

	float* ux_d = (float*)malloc(sizeof(float));
	float* uy_d = (float*)malloc(sizeof(float));
	float* uz_d = (float*)malloc(sizeof(float));
	float* rho_d = (float*)malloc(sizeof(float));

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("Compute capability: %d.%d\n", devProp.major, devProp.minor);

//***** Memory allocated on device *******

	cudaMalloc((void**)&f_d, field_size);
	cudaMemcpy(f_d, f, field_size, cudaMemcpyHostToDevice);    // copying distributions to device

	cudaMalloc((void**)&ux_d, sizeof(float));
	cudaMalloc((void**)&uy_d, sizeof(float));
	cudaMalloc((void**)&uz_d, sizeof(float));
	cudaMalloc((void**)&rho_d, sizeof(float));


	float* uxt = (float*)malloc(sizeof(float));     // Velocity at any point to be printed on screen
	float* rhot = (float*)malloc(sizeof(float));    // Density at any point to be printed on screen
	float* dev_uxt, * dev_rhot;  // Device memory

	cudaMalloc((void**)&dev_uxt, sizeof(float));
	cudaMalloc((void**)&dev_rhot, sizeof(float));

	cudaMemcpyToSymbol(dev_w, w, NUMSPD * sizeof(float));           // copying weight functions to device
	cudaMemcpyToSymbol(dev_ex, ex, NUMSPD * sizeof(float));         // copying lattice velocity to device
  cudaMemcpyToSymbol(dev_ey, ey, NUMSPD * sizeof(float));
	cudaMemcpyToSymbol(dev_ez, ez, NUMSPD * sizeof(float));
  cudaMemcpyToSymbol(dev_kb, kb, NUMSPD * sizeof(float));
	//dim3 BLOCKS(TPB2D, TPB2D, 1);                                           // no. of threads per block
	//dim3 GRIDS((NX + TPB2D - 1) / TPB2D, (NY + TPB2D - 1) / TPB2D, NZ);     // no. of blocks per grid
 
int BLOCK = 512;       
int GRID = ceil(NX*NY*NZ/512);  

int BLOCKS = 256;        //works till blocksize = 256 //512
int GRIDS = ceil(NX*NY*NZ/256);  //1

	double comp_collide_time = 0;
	double comp_stream_time = 0;
  double total_time=0;

	for (int t = 0; t <= 151; t++) {

		start1 = clock();
		collideLBGK << <GRIDS, BLOCKS >> > (f_d,ux_d,rho_d);
		cudaThreadSynchronize();
		finish1 = clock();

		start2 = clock();
 
 // streaming in forward direction //
   
   for(int i=0;i<NX;i++)
   {
		stream_Xforward << <GRID, BLOCK >> > (f_d,i);
		cudaThreadSynchronize();
   }
   
   for(int j=0;j<NY;j++)
   {
		stream_Yforward << <GRID, BLOCK >> > (f_d,j);
		cudaThreadSynchronize();
   }
   for(int k=0;k<NZ;k++)
   {
		stream_Zforward << <GRID, BLOCK >> > (f_d,k);
		cudaThreadSynchronize();
   }
   
 // streaming in backward direction //
 
   for(int i=NX-1;i>=0;i--)
   {
    stream_Xbackward << <GRID, BLOCK >> > (f_d,i);
		cudaThreadSynchronize();
   }
   for(int j=NY-1;j>=0;j--)
   {
    stream_Ybackward << <GRID, BLOCK >> > (f_d,j);
		cudaThreadSynchronize();
   }  
   for(int k=NZ-1;k>=0;k--)
   {
    stream_Zbackward << <GRID, BLOCK >> > (f_d,k);
		cudaThreadSynchronize();
   }
   
		finish2 = clock();

 // computing time //
 
		comp_collide_time += (double)(finish1 - start1);
		comp_stream_time += (double)(finish2 - start2);

		if (t % 1 == 0) {
			Watch << <1, 1 >> > (rho_d, ux_d, dev_uxt, dev_rhot);   // Obtaining velocity and density at a particular point from the array values on device
			cudaMemcpy(uxt, dev_uxt, sizeof(float), cudaMemcpyDeviceToHost);    // copying value at a point to host
			cudaMemcpy(rhot, dev_rhot, sizeof(float), cudaMemcpyDeviceToHost);
    OutWatch(t, *uxt, *rhot);   // printing on screen
		}

	}

 

  total_time=(comp_collide_time+comp_stream_time)/CLOCKS_PER_SEC;
  
  printf("Time Take for comp_collide function : %7.2lfs\n", (double)(comp_collide_time) / CLOCKS_PER_SEC);
	printf("Time Take for comp_stream function : %7.2lfs\n", (double)(comp_stream_time) / CLOCKS_PER_SEC);

  printf("Time Take total : %7.2lfs\n", (double)(total_time) );
  printf("MLUPS :  %.2lf\n",((double)NNODES*pow(10,-6)*(double)151)/(total_time));

  
	cudaMemcpy(f, f_d, field_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ux, ux_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, uy_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(uz, uz_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho, rho_d, sizeof(float), cudaMemcpyDeviceToHost);



	Outp(rho, ux, uy, uz, f);

	
  
  	//free(rhoav);
	cudaFree(f_d);
	cudaFree(ux_d);
	cudaFree(uy_d);
	cudaFree(uz_d);
	cudaFree(rho_d);
	//cudaFree(dev_rhoav);
	//cudaFree(dev_domain);
  
    free(f);
  
	return 0;
}


//****** Conversion to row major on host (3D to 1D array)  ******
int Ord3(int x, int y, int z, int nx, int ny, int nz) {
	return  x + y * nx + z * nx * ny;
}

//****** Conversion to row major on device (3D to 1D array)  ******
__device__ int d_Ord3(int x, int y, int z, int nx, int ny, int nz) {
	return x + y * nx + z * nx * ny;
}

__device__ void d_Ord3r(int id, int* x, int* y, int* z, int ny, int nx) {
	*x = id % nx;			//chaNGES MAde
	*y = ((id - *x) / nx) % ny;
	*z = (((id - *x) / nx) - *y) / ny;
	return;
}

void OutWatch(int t, float uxt, float rhot) {
	printf("Time Step %d, ", t);
	printf("ux[20][25][25]= %f  Density = %lf\n", uxt, rhot);
}

//****** Initialization of distributions on host ******

void Initialize(float* f)
{
    float ux=0.,uy=0.,uz=0.;
    float rho0=1.0;
	for (int i = 0; i < NX; i++)
    {
		for (int j = 0; j < NY; j++)
        {
			for (int k = 0; k < NZ; k++)
			{
                ux = U0*sin(i*(2*PI/NX))*(cos(3*j*(2*PI/NY))*cos(k*(2*PI/NZ)) - cos(j*(2*PI/NY))*cos(3*k*(2*PI/NZ)));
                uy = U0*sin(j*(2*PI/NY))*(cos(3*k*(2*PI/NZ))*cos(i*(2*PI/NX)) - cos(k*(2*PI/NZ))*cos(3*i*(2*PI/NX)));
                uz = U0*sin(k*(2*PI/NZ))*(cos(3*i*(2*PI/NX))*cos(j*(2*PI/NY)) - cos(i*(2*PI/NX))*cos(3*j*(2*PI/NY)));

                for(int a=0;a<NUMSPD;a++)
                {
                    float t =  ex[a]*ux + ey[a]*uy + ez[a]*uz;
                    float u2 = ux*ux + uy*uy + uz*uz;
                    f[NUMSPD * Ord3(i, j, k, NX, NY, NZ) + a] = w[a]*rho0*(1 + 3*t + 4.5*t*t - 1.5*u2);
                }
			}
		}
	}
}


void Outp(const float* rho, const float* ux, const float* uy, const float* uz, const float* f) {


	FILE* fp = fopen(OUTPUTFILENAME, "w");
	fprintf(fp, "Title=\"LBM Poisuille Flow\"\n");
	fprintf(fp, "VARIABLES=\"X\",\"Y\",\"Z\",\"Ux\",\"Uy\",\"Uz\",\"rho\"\n");
	fprintf(fp, "ZONE T=\"BOX\",I=%d,J=%d,K=%d,F=POINT\n", NX, NY, NZ);
	for (int k = 0; k < NZ; k++) {
		fprintf(fp, "%d\t ", (int)k);
		fprintf(fp, "%0.8f\t ", ux[Ord3(20, 25, k, NX, NY, NZ)]);
		fprintf(fp, "%0.8f\t ", uy[Ord3(20, 25, k, NX, NY, NZ)]);
		fprintf(fp, "%0.8f\t ", uz[Ord3(20, 25, k, NX, NY, NZ)]);
		fprintf(fp, "%0.8f\t ", rho[Ord3(20, 25, k, NX, NY, NZ)]);
		/*for (int a = 0; a < NUMSPD; a++) {
			fprintf(fp, "%0.8f\t ", f[NUMSPD * Ord3(20, 25, k, NX, NY, NZ) + a]);
		}*/
		fprintf(fp, "\n");
	}
	fclose(fp);
	return;
}

__global__ void Watch(float* rho_d,float* ux_d, float* uxt, float* rhot) {

	*uxt = *ux_d;
	*rhot = *rho_d;

	return;
}

//****** Collision ******

__global__ void collideLBGK(float* f_d, float* ux_d, float* rho_d) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    
//printf("working\n");
  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

	float ux_r=0.,uy_r=0.,uz_r=0.;
	float rho_r=0.;
	float feq_r[NUMSPD];

	if ((X < NX) && (Y < NY) && (Z < NZ)) {
		//int nnodes = NX * NY * NZ;
		int tid = X + Y * NX + Z * NX * NY;
   


		      //*** macroscopic variables ***/

#pragma unroll
		for (int spd = 0; spd < NUMSPD; spd++)
        {
            rho_r += f_d[NUMSPD * tid + spd];
			ux_r += dev_ex[spd] * f_d[NUMSPD * tid + spd];
			uy_r += dev_ey[spd] * f_d[NUMSPD * tid + spd];
			uz_r += dev_ez[spd] * f_d[NUMSPD * tid + spd];
		}

		ux_r /= rho_r;
		uy_r /= rho_r;
		uz_r /= rho_r;

		if( X==20 && Y==25 && Z==25)
        {
            *ux_d=ux_r;
            *rho_d=rho_r;
        }

            //*** equilibrium function ***

    float term1,term2;
    float u2=ux_r*ux_r+uy_r*uy_r+uz_r*uz_r;
#pragma unroll

    for(int spd=0;spd<NUMSPD;spd++)
        {
            term1 = ux_r*dev_ex[spd] + uy_r*dev_ey[spd] +uz_r*dev_ez[spd];
            term2 = term1*term1;
            feq_r[spd] = dev_w[spd]*rho_r*(1.0 + 3.0*term1 + 4.5*term2 - 1.5*u2);
        }

              //*** collision step ***
              
#pragma unroll
		for (int spd = 0; spd < NUMSPD; spd++) {
			f_d[NUMSPD * tid + spd] = f_d[NUMSPD * tid + spd] - (f_d[NUMSPD * tid + spd] - feq_r[spd]) / (TAU);
		}
	}
}



            //****** Streaming ******
            

//***** streaming X distributions *****

__global__ void stream_Xforward(float* f_d,const int i) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(i==X)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if ( a==2 || a==9 || a==10 || a==13 || a==14 || a==23 || a==24 || a==25 || a==26 )
       {
         int xn = (X + dev_ex[a ] + NX) % NX;		//Periodic boundary conditions included
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];

      }
    }
   }
 }
}
__global__ void stream_Xbackward(float* f_d,const int i) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(i==X)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if ( a==1 || a==19 || a==20 || a==7 || a==8 || a==11 || a==12 || a==21 || a==22 )
       {
         int xn = (X + dev_ex[a ] + NX) % NX;		
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];
      }
    }
   }
 }
}



//***** streaming Y distributions *****

__global__ void stream_Yforward(float* f_d,const int j) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(j==Y)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if (a==4 ||  a==17 || a==18)
       {
         int xn = (X + dev_ex[a ] + NX) % NX;		
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];

      }
    }
   }
 }
}
__global__ void stream_Ybackward(float* f_d,const int j) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(j==Y)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if (a==3 || a==15 || a==16)
       {
         int xn = (X + dev_ex[a ] + NX) % NX;		
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];
      }
    }
   }
 }
}


//***** streaming Z distributions *****

__global__ void stream_Zforward(float* f_d,const int k) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(k==Z)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if ( a==6 )
       {
         int xn = (X + dev_ex[a ] + NX) % NX;	
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];

      }
    }
   }
 }
}
__global__ void stream_Zbackward(float* f_d,const int k) 
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;    

  int X = idx / (NZ * NY);
          idx -= (X * NZ * NY);
  int Y = idx / NZ;
  int Z = idx % NZ;

 if(k==Z)
 {
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) 
   {

#pragma unroll
		for (int a = 0; a < NUMSPD; a++) 
    {
       if (a==5)
       {
         int xn = (X + dev_ex[a ] + NX) % NX;		
			int yn = (Y + dev_ey[a ] + NY) % NY;
			int zn = (Z + dev_ez[a ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + a] = f_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + a];
      }
    }
   }
 }
}


