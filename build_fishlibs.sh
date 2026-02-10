nvcc plugins/cuda/karlsen-cuda-native/src/karlsen-cuda.cu -std=c++14 -O3 --restrict --ptx --gpu-architecture=compute_86 --gpu-code=sm_86 -o plugins/cuda/resources/karlsen-cuda-sm86.ptx -Xptxas -O3 -Xcompiler -O3

nvcc plugins/cuda/karlsen-cuda-native/src/karlsen-cuda.cu -std=c++14 -O3 --restrict --ptx --gpu-architecture=compute_75 --gpu-code=sm_75 -o plugins/cuda/resources/karlsen-cuda-sm75.ptx -Xptxas -O3 -Xcompiler -O3

nvcc plugins/cuda/karlsen-cuda-native/src/karlsen-cuda.cu -std=c++14 -O3 --restrict --ptx --gpu-architecture=compute_61 --gpu-code=sm_61 -o plugins/cuda/resources/karlsen-cuda-sm61.ptx -Xptxas -O3 -Xcompiler -O3
