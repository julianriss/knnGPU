__global__ void setData(float *dataset, float *network) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	network[tid] = dataset[tid];
}

__global__ void setTarget(float *target, int digit) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == digit) {
		target[tid] = 1;
	} else {
		target[tid] = 0;
	}
}

__global__ void fcForward(float *input, float *weights, float *result, int k,
		int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;
	if (tid < n) {
		for (int i = 0; i < k; i++)
			sum += input[i] * weights[(i * n) + tid];
		result[tid] += sum;
	}
}

__global__ void cvForward(float *input, float *mask, float *result,
		int imgWidth, int kwidth) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;

	for (int i = 0; i < kwidth; i++) {
		for (int e = 0; e < kwidth; e++) {
			sum += input[imgWidth * (e + blockIdx.x) + (i + threadIdx.x)]
					* mask[kwidth * (kwidth - e) + (kwidth - i)];
		}
	}
	result[id] = sum;
}

__global__ void cvBackward(float *input, float *mask, float *result, float *map,
		int imgWidth, int kwidth) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;

	for (int i = 0; i < kwidth; i++) {
		for (int e = 0; e < kwidth; e++) {
			sum += input[imgWidth * (e + blockIdx.x) + (i + threadIdx.x)]
					* mask[kwidth * e + i];
		}
	}
	result[id] = sum * map[id] * (1.0 - map[id]); /////////////////////////
}

__global__ void resetmap(float *result) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	result[tid] = 0.0;
}

__global__ void sigmoid(float *result) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	result[tid] = 1.0 / (1.0 + exp(-result[tid]));
}

__global__ void outputdeltaGPU(float *delta, float *output, float *target) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	delta[tid] = (target[tid] - output[tid]) * output[tid]
			* (1.0 - output[tid]);
}

__global__ void fcBackpropagation(float *outputdelta, float *outputweights,
		float *hiddendelta, float *hidden, int hSize, int oSize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float sum = 0;
	if (tid < hSize) {
		for (int i = 0; i < oSize; i++) {
			sum += outputdelta[i] * outputweights[(i * hSize) + tid];
		}
		hiddendelta[tid] = sum * hidden[tid] * (1.0 - hidden[tid]); ////////////////////////////
	}
}

__global__ void fcUpdate(float *outputweights, float *outputdelta,
		float *hidden) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	outputweights[tid] += 0.1 * hidden[blockIdx.x] * outputdelta[threadIdx.x];
}

__global__ void cvUpdate(float *deltabuffer, float *delta, float *input,
		int imgWidth, int kwidth) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < kwidth; i++) {
		for (int e = 0; e < kwidth; e++) {
			deltabuffer[kwidth * e + i] += 0.1
					* input[imgWidth * (e + blockIdx.x) + (i + threadIdx.x)]
					* delta[id];
		}
	}
}

__global__ void cvAdd(float *deltabuffer, float *weights) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	weights[id] += deltabuffer[id];
}

__global__ void zeropadding(float *deltazeropadded, float *delta, int imgWidth,
		int kwidth) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x <= kwidth - 2 || blockIdx.x <= kwidth - 2
			|| threadIdx.x >= imgWidth + (kwidth - 1)
			|| blockIdx.x >= imgWidth + (kwidth - 1)) {
		deltazeropadded[id] = 0.0;

	} else {
		deltazeropadded[id] = delta[imgWidth * (blockIdx.x - (kwidth - 1))
				+ (threadIdx.x - (kwidth - 1))];
	}
}

__global__ void maxForward(float *result, float *resultpool, int imgWidth) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float sum = result[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)];

	if (result[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] > sum) {
		sum = result[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)];
	}

	if (result[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] > sum) {
		sum = result[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)];
	}

	if (result[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] > sum) {
		sum = result[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)];
	}

	resultpool[tid] = sum;

}

__global__ void maxBackward(float *map, float *deltares, float *deltasrc,
		int imgWidth) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float sum = map[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)];
	deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)] = deltasrc[tid];

	deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] = 0.0;
	deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] = 0.0;
	deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] = 0.0;

	if (map[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] > sum) {
		sum = map[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)];
		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] =
				deltasrc[tid];

		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] = 0.0;

	}

	if (map[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] > sum) {
		sum = map[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)];
		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] =
				deltasrc[tid];

		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] = 0.0;

	}

	if (map[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] > sum) {
		sum = map[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)];
		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2 + 1)] =
				deltasrc[tid];

		deltares[imgWidth * (blockIdx.x * 2 + 1) + (threadIdx.x * 2)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2 + 1)] = 0.0;
		deltares[imgWidth * (blockIdx.x * 2) + (threadIdx.x * 2)] = 0.0;

	}

}
