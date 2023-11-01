#include <torch/extension.h>
//#include <torch/serialize/tensor.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <thrust/device_vector.h>
//#include <thrust/sort.h>

#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

#ifdef __cplusplus
    extern "C" {
#endif




__global__ void cost_kernel_forward(const int n, const float* x, const float* y, const int shift1, const int shift2, const int stride1, const int stride2, const int heightx, const int widthx, const int heighty, const int widthy, const int channel,  float* top_data) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int stepx = heightx * widthx;
    int stepy = heighty * widthy;
    int size1 = (shift1*2+1);
    int size2 = (shift2*2+1);
    int loc0 = index/(size1*size2*stepx);
    int loc34 = index%stepx;
    int loc1 = index/(size2*stepx)%size1-shift1;
    int loc2 = index/stepx%size2-shift2;
    
    int cur_x = loc0*stepx*channel + loc34;
    int row = loc34/widthx/stride1;
    int col = (loc34%widthx)/stride2;
    if(row+loc1<0||row+loc1>=heighty||col+loc2<0 ||col+loc2>=widthy){
        top_data[index]=0;
        return;
    }
    int cur_y = loc0*stepy*channel + (row+loc1)*widthy + col+loc2;
    float temp = 0;
    for(int i=0;i<channel;i++)
        temp += (x[cur_x+i*stepx]*y[cur_y+i*stepy]);
    top_data[index]=temp;
    

}



__global__ void cost_backward_x(const int n, const float* y, const float* top_diff, const int shift1, const int shift2, const int stride1, const int stride2, const int heightx, const int widthx, const int heighty, const int widthy, const int channel, float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (index >= n) {
        return;
    }
    int stepx = heightx * widthx;
    int stepy = heighty * widthy;
    int size1 = (shift1*2+1);
    int size2 = (shift2*2+1);
    int loc0 = index/(channel*stepx);
    int loc34 = index%stepx;
    int rowx = loc34/widthx;
    int colx = loc34%widthx;
    int rowy = rowx/stride1;
    int coly= colx/stride2;
    int basey = index/stepx*stepy;
    int base = loc0*size1*size2*stepx + loc34;
    float temp = 0;
    for(int i=0;i<size1;i++){
        int ry = rowy + i - shift1;
        if(ry<0 || ry>=heighty)
            continue;
        int basei = base + i*size2*stepx;
        int baseyi = basey + ry * widthy;
        for(int j=0;j<size2;j++){
            int cy = coly + j - shift2;
            if(cy<0||cy>=widthy)
                continue;
            temp += top_diff[basei+j*stepx] * y[baseyi+cy];
        }
    }
    bottom_diff[index]=temp;
}
__global__ void cost_backward_y(const int n, const float* x, const float* top_diff, const int shift1, const int shift2, const int stride1, const int stride2, const int heightx, const int widthx, const int heighty, const int widthy, const int channel, float* bottom_diff){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (index >= n) {
        return;
    }
    int stepx = heightx * widthx;
    int stepy = heighty * widthy;
    int size1 = (shift1*2+1);
    int size2 = (shift2*2+1);
    int loc0 = index/(channel*stepy);
    int loc34 = index%stepy;
    int rowy = loc34/widthy;
    int coly = loc34%widthy;
    int rowx = rowy*stride1;
    int colx= coly*stride2;
    int basex = index/stepy*stepx;
    int base = loc0*size1*size2*stepx;
    float temp = 0;
    for(int i=-shift1*stride1;i<(shift1+1)*stride1;i++){
        int rx = rowx + i;
        if(rx<0||rx>=heightx)
            continue;
        int loc1 = -i/stride1 + shift1;
        int basei = base + loc1*size2*stepx + rx*widthx;
        int basexi = basex + rx * widthx;
        for(int j=-shift2*stride2;j<(shift2+1)*stride2;j++){
            int cx = colx + j;
            if(cx<0||cx>=widthx)
                continue;
            int loc2 = -j/stride2 + shift2;
            int curx = basexi + cx;
            int cur = basei + loc2*stepx +cx;
            temp += top_diff[cur] * x[curx];
        }
    }
    bottom_diff[index]=temp;
}
            
        
    

void cost_volume_forward (at::Tensor input1, at::Tensor input2,
                          at::Tensor output, const int shift1=48,
                          const int shift2 = 48, const int stride1=1,
                          const int stride2=1){

	int num = input1.size(0);
	int channel = input1.size(1);
	int heightx = input1.size(2);
	int widthx = input1.size(3);
	int heighty = input2.size(2);
	int widthy = input2.size(3);

	float *cost = output.data<float>();

	const float *x = input1.data<float>();
	const float *y = input2.data<float>();

	int n = output.numel();
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
//        printf("%d %d %d %d %d %d %d %d\n", num, channel, height, width, wsize, n, threads, N);
        cost_kernel_forward <<<threads, CUDA_NUM_THREADS>>>(n, x, y, shift1, shift2, stride1, stride2, heightx, widthx, heighty, widthy, channel,  cost);
//        printf("sgf down done...\n");
}
void cost_volume_backward (at::Tensor input1, at::Tensor input2,
                          at::Tensor grad_output, at::Tensor grad_input1, at::Tensor grad_input2,
                          const int shift1=48, const int shift2 = 48, const int stride1=1,
                          const int stride2=1){

	int num = input1.size(0);
	int channel = input1.size(1);
	int heightx = input1.size(2);
	int widthx = input1.size(3);
	int heighty = input2.size(2);
	int widthy = input2.size(3);

        const float *grad_out = grad_output.data<float>();
	const float *x = input1.data<float>();
	const float *y = input2.data<float>();

	float *gradx = grad_input1.data<float>();
	float *grady = grad_input2.data<float>();

	int n = input1.numel();
	int threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
//        printf("%d %d %d %d %d %d %d %d\n", num, channel, height, width, wsize, n, threads, N);
        cost_backward_x <<<threads, CUDA_NUM_THREADS>>>(n, y, grad_out, shift1, shift2, stride1, stride2, heightx, widthx, heighty, widthy, channel,  gradx);
	n = input2.numel();
	threads = (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
        cost_backward_y <<<threads, CUDA_NUM_THREADS>>>(n, x, grad_out, shift1, shift2, stride1, stride2, heightx, widthx, heighty, widthy, channel,  grady);
//        printf("sgf down done...\n");
}



 
#ifdef __cplusplus
    }
#endif
