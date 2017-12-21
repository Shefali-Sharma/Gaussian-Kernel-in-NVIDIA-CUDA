/*  Shefali
 *  Sharma
 *  sharma92
 */
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <iostream>
#ifndef A3_HPP
#define A3_HPP
#define NUM_THREADS 1024
#define PI 3.14

#define CONST(n,h) 1.0/sqrt(2.0*PI)/(float)n/h
#define calcK(a,b,h) ((a-b)*(a-b))/2/h/h

using namespace std;

void PrintSum(const vector<float>& AR1, const vector<float>& AR2) {
    
    cout<<endl;
    float sum1 = 0, sum2 = 0;
    for (int i = 0; i < AR1.size(); i++) {
        cout<<AR1[i]<<"        "<<AR2[i]<<endl;
        sum1 += AR1[i]; sum2 += AR2[i];
    }
    //cout<<"Sum "<<sum1<<" "<<sum2<<endl;
    
}

__global__ void Kernel(float *d_x , float *d_y , int N, float h)
{
    __shared__ float *sharedVar;
    
    int I_dx = threadIdx.x + blockIdx.x * blockDim.x;
    
    sharedVar[I_dx] = d_x[I_dx];
    
    //Sync in threads
    __syncthreads();
    
    float sum = 0;
    if( I_dx < N)
    {
        for(int j = 0; j < N; j++)
        {
            sum = sum + (( 1 / sqrt(2 * PI)) * exp ( - ( ( ( d_x[I_dx] - sharedVar[j] ) / h ) * (d_x[I_dx] - sharedVar[j]) / 2) ) );
        }
        d_y[I_dx] = sum / (N * h);
    }
}

__global__ void getK(float *d_x , float *d_y , int N, float h, int i)
{
    int I_dx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //__shared__ float arr[N];
    
    //float d_xy[Idx] = (( 1 / sqrt(2 * PI)) * exp ( - ( ( pow ( ( ( d_x[i] - d_x[I_dx] ) / h ), 2) ) / 2) ) );
    
    //arr[I_dx] = K;
    
    
    __syncthreads();
    
    
    
}

void seqValuesForTest(int n, float h, const std::vector<float>& x, std::vector<float>& y){
    vector<float> test_vec;
    
    for(int i = 0; i < n; i ++) {
        float par_val = 0.0f;
        for (int j = 0; j < n; j++) {
            par_val += CONST(n,h)*(float)exp(-calcK(x[i], x[j], h));
        }
        
        test_vec.push_back(par_val);
    }
    
    PrintSum(test_vec,y);
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    
    //seqValuesForTest(n, h, x, y);
    std::vector<float> res(n, 0.0);
    
    int NumBlocks = ((NUM_THREADS + n - 1)/NUM_THREADS);
    
    float *d_x = NULL;
    float *d_y = NULL;
    
    cudaMalloc(&d_x,sizeof(float)* n);
    cudaMalloc(&d_y,sizeof(float)* n);
    
    cudaMemcpy(d_x, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    
    dim3 d_NUM_THREADS(NUM_THREADS);
    dim3 d_NumBlocks(NumBlocks);
    
    int NumThr = n;
    
    Kernel<<<d_NumBlocks , d_NUM_THREADS>>>(d_x , d_y , n, h);
    cudaMemcpy(y.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    //PrintSum(x,y);
    
    //vector1.insert( vector1.end(), vector2.begin(), vector2.end() );
    /*
     for(int i=0; i<NUM_THREADS; i++){
     
     //Kernel<<<1 , d_NUM_THREADS>>>(d_x , d_y , n, h, i);
     //cudaMemcpy(y.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
     if((NumThr - N)>1024){
     getK <<<1,  1024 >>>(d_x, d_y , n, h, i);
     NumThr = NumThr - N;
     }
     else{
     getK <<<1,  NumThr >>>(d_x, d_y , n, h, i);
     NumThr = NumThr - N;
     }
     
     cudaMemcpy(y.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
     
     }
     */
    
    cudaFree(d_x);
    cudaFree(d_y);
    
} // gaussian_kde

#endif // A3_HPP
