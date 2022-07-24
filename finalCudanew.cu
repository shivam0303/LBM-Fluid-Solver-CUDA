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
#include <fstream>

using namespace std;


#define NX 100 	                        //Number of grid points along the length
#define NY 100	                        //Number of grid points along the height
#define NZ 100
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

__global__ void collideLBGK(float* f_d,float* fnew_d,float* ux_d, float* rho_d);

__global__ void stream(float* f_d,const float* fnew_d);


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
	float* f = (float*)malloc(field_size);                                    // post coll. distribution
        

	float* ux = (float*)malloc(sizeof(float));
	float* uy = (float*)malloc(sizeof(float));
	float* uz = (float*)malloc(sizeof(float));
	float* rho = (float*)malloc(sizeof(float));



    Initialize(f);  // Initializing distributions on host



//****** Memory allocation for device ******

	float* f_d = (float*)malloc(field_size);
    float* fnew_d = (float*)malloc(field_size);    

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
    cudaMalloc((void**)&fnew_d, field_size);

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
	dim3 BLOCKS(TPB2D, TPB2D, 1);                                           // no. of threads per block
	dim3 GRIDS((NX + TPB2D - 1) / TPB2D, (NY + TPB2D - 1) / TPB2D, NZ);     // no. of blocks per grid

	double comp_collide_time = 0;
	double comp_stream_time = 0;
    double total_time=0;

	for (int t = 0; t <= 151; t++) {

		start1 = clock();
		collideLBGK << <GRIDS, BLOCKS >> > (f_d,fnew_d,ux_d,rho_d);
		cudaDeviceSynchronize();
		finish1 = clock();

		
		start2 = clock();
		stream << <GRIDS, BLOCKS >> > (f_d, fnew_d);
		cudaDeviceSynchronize();
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

  
  	//free(rhoav);
	cudaFree(f_d);
	cudaFree(ux_d);
	cudaFree(uy_d);
	cudaFree(uz_d);
	cudaFree(rho_d);
  
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


__global__ void Watch(float* rho_d,float* ux_d, float* uxt, float* rhot) {

	*uxt = *ux_d;
	*rhot = *rho_d;

	return;
}

//****** Collision ******

__global__ void collideLBGK(float* f_d, float* fnew_d, float* ux_d, float* rho_d) {
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	int Z = threadIdx.z + blockIdx.z * blockDim.z;

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
			fnew_d[NUMSPD * tid + spd] = f_d[NUMSPD * tid + spd] - (f_d[NUMSPD * tid + spd] - feq_r[spd]) / (TAU);
		}
	}
}



            //****** Streaming ******
            

//***** streaming X distributions *****

__global__ void stream(float* f_d, const float* fnew_d) {

	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;
	int Z = threadIdx.z + blockIdx.z * blockDim.z;
	if ((X >= 0) && (Y >= 0) && (Z >= 0) && (X < NX) && (Y < NY) && (Z < NZ)) {
		
#pragma unroll
		for (int spd = 0; spd < NUMSPD; spd++) {
			int xn = (X + dev_ex[spd ] + NX) % NX;		//Periodic boundary conditions included
			int yn = (Y + dev_ey[spd ] + NY) % NY;
			int zn = (Z + dev_ez[spd ] + NZ) % NZ;
			f_d[NUMSPD * (xn + yn * (NX)+zn * (NX) * (NY)) + spd] = fnew_d[NUMSPD * (X + Y * (NX)+Z * (NX) * (NY)) + spd];

		}
	}
}


