#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <Windows.h>

using namespace std;

const int epochs = 700;
const int parameterCount = 5;
const bool display = true;

inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
	}

	return result;
}

__global__ void kernel(float *particles, int particleCount)
{
	extern __shared__ float sharedParticles[];

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float x = particles[parameterCount * index];
	float y = particles[parameterCount * index + 1];
	float mass = particles[parameterCount * index + 2];
	float vx = particles[parameterCount * index + 3];
	float vy = particles[parameterCount * index + 4];

	sharedParticles[parameterCount * index] = x;
	sharedParticles[parameterCount * index + 1] = y;
	sharedParticles[parameterCount * index + 2] = mass;
	sharedParticles[parameterCount * index + 3] = vx;
	sharedParticles[parameterCount * index + 4] = vy;
	__syncthreads();

	float G = 0.05;
	float dt = 0.2;
	float e = 0.5;
	
	for (int i = 0; i < epochs; i++) {
		float fx = 0;
		float fy = 0;

		for (int j = 0; j < particleCount; j++) {
			if (j != index) {
				float jx = sharedParticles[parameterCount * j];
				float jy = sharedParticles[parameterCount * j + 1];
				float jmass = sharedParticles[parameterCount * j + 2];

				if (x != jx) {
					fx = fx + (G * mass * jmass * -(x - jx)) / pow(pow(abs(x - jx), 2) + e*e, 3 / 2);
				}
				if (y != jy) {
					fy = fy + (G * mass * jmass * -(y - jy)) / pow(pow(abs(y - jy), 2) + e*e, 3 / 2);
				}
			}
		}

		vx = vx + fx / mass * dt;
		vy = vy + fy / mass * dt;

		x = x + vx * dt;
		y = y + vy * dt;

		if (display) {
			//printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", i, index, x, y, fx, fy, vx, vy);
			printf("%f\t%f\n", x, y);
		}

		__syncthreads();

		sharedParticles[parameterCount * index] = x;
		sharedParticles[parameterCount * index + 1] = y;
		sharedParticles[parameterCount * index + 2] = mass;
		sharedParticles[parameterCount * index + 3] = vx;
		sharedParticles[parameterCount * index + 4] = vy;
		__syncthreads();
	}
}

int main()
{
	srand((int)time(nullptr));

	bool random = true;

	const int particlesSize = 3;
	float particles[parameterCount * particlesSize];

	if (random) {
		for (int r = 0; r < particlesSize; r++) {
			particles[parameterCount * r] = (float)((rand() % 10000) / 100.0);
			particles[parameterCount * r + 1] = (float)((rand() % 10000) / 100.0);
			particles[parameterCount * r + 2] = (float)((rand() % 10000) / 100.0);
			particles[parameterCount * r + 3] = (float)((rand() % 200 - 100) / 100.0);
			particles[parameterCount * r + 4] = (float)((rand() % 200 - 100) / 100.0);
		}
	}
	else {
		// syntax: {x, y, mass}
		float newParticles[] = {
			10, 25, 140, 0, -0.5,
			25, 45, 10, 0, 2,
			-35, 35, 90.3, 0, 1.1
			//50, 70, 14.2,
			//35, 40, 35.4
		};
		copy(newParticles, newParticles + sizeof(newParticles)/sizeof(float), particles);
	}

	if (display) {
		cout << "index\tx\t\ty\t\tmass\tvx\tvy" << endl;
	}

	if (display) {
		for (int i = 0; i < particlesSize; i++) {
			printf("%d\t%f\t%f\t%f\t%f\t%f\n", 
				i, 
				particles[parameterCount * i], 
				particles[parameterCount * i + 1], 
				particles[parameterCount * i + 2], 
				particles[parameterCount * i + 3], 
				particles[parameterCount * i + 4]
			);
		}
	}

	if (display) {
		cout << endl << "epoch\tindex\tx\t\ty\t\tfx\t\tfy\t\tvx\t\tvy" << endl;
	}

	float endParticles[2][3];

	double tt, tu;
	LARGE_INTEGER ti, to, tb, te, tk, tm, tf;
	QueryPerformanceFrequency(&tf);

	dim3 threadsPerBlock(particlesSize);
	dim3 blocksPerGrid(1);

	float *cudaParticles;

	QueryPerformanceCounter(&tb);

	checkCuda(cudaMalloc((void**)&cudaParticles, sizeof(particles)));
	checkCuda(cudaMemcpy(cudaParticles, particles, sizeof(particles), cudaMemcpyHostToDevice));

	QueryPerformanceCounter(&tk);
	kernel << <blocksPerGrid, threadsPerBlock, particlesSize*parameterCount*4 >> >(cudaParticles, particlesSize);
	QueryPerformanceCounter(&tm);

	checkCuda(cudaMemcpy(endParticles, cudaParticles, sizeof(particles), cudaMemcpyDeviceToHost));
	cudaFree(cudaParticles);

	QueryPerformanceCounter(&te);
	tu = 1000.0*(double(tm.QuadPart - tk.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU time: " << tu << " ms\n";
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU time with memory operations: " << tt << " ms\n";

	checkCuda(cudaDeviceReset());

	cout << "DONE";
	cin.ignore();

    return 0;
}
