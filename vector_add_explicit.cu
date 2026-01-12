#include <cuda_runtime.h>
#include <stdio.h>

/*
 * CUDA KERNEL - Runs on GPU
 * - Launched with <<<blocks, threads>>> syntax
 * - Each thread processes one element
 * - Must use __global__ keyword
 */
__global__ void vectorAdd(const float *array_a, const float *array_b, float *array_c, size_t element) 
{
    // Calculate unique thread ID across all blocks
    // blockIdx.x = which block am I in?
    // threadIdx.x = which thread am I within my block?
    // blockDim.x = how many threads per block?
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Boundary check: some threads may be beyond array size
    if (i < element)
    {
        array_c[i] = array_a[i] + array_b[i];
    }
}

int main ()
{
    // ====================================
    // STEP 1: PROBLEM SETUP
    // ====================================
    size_t element = 1 << 24;  // Bit shift: 2^24 = 16,777,216 elements
    size_t bytes = sizeof(float) * element;  // 4 bytes * 16M = 64 MB

    printf("Vector Addition - Explicit Memory\n");
    printf("Array size: %zu elements (%.2f MB)\n", element, bytes/1024.0/1024.0); 
    
    // ====================================
    // STEP 2: ALLOCATE HOST (CPU) MEMORY
    // ====================================
    // Host memory = RAM on CPU
    // These pointers point to CPU memory
    float* host_array_a = new float[element];
    float* host_array_b = new float[element];
    float* host_array_c = new float[element];  // Will store results
    
    // ====================================
    // STEP 3: INITIALIZE DATA ON HOST
    // ====================================
    // We must initialize data on CPU before sending to GPU
    printf("Initializing arrays...\n");
    for (size_t i = 0; i < element; i++) {
        host_array_a[i] = 1.0f;
        host_array_b[i] = 2.0f;
    }
    
    // ====================================
    // STEP 4: ALLOCATE DEVICE (GPU) MEMORY
    // ====================================
    // Device memory = VRAM on GPU
    // cudaMalloc allocates memory on GPU
    // Note: We pass ADDRESS of pointer (&d_a) so cudaMalloc can modify it
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);  // Allocate 64 MB on GPU for array A
    cudaMalloc(&d_b, bytes);  // Allocate 64 MB on GPU for array B
    cudaMalloc(&d_c, bytes);  // Allocate 64 MB on GPU for result C
    
    // KEY CONCEPT: At this point we have TWO separate memory spaces:
    // - host_array_* pointers → CPU RAM
    // - d_* pointers → GPU VRAM
    // They are NOT the same memory!
    
    // ====================================
    // STEP 5: SETUP TIMING EVENTS
    // ====================================
    // cudaEvent = GPU-aware timers (more accurate than CPU timers)
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    
    // ====================================
    // STEP 6: TRANSFER DATA HOST → DEVICE
    // ====================================
    // cudaMemcpy copies data between CPU and GPU
    // This is SLOW because data crosses PCIe bus
    float h2d_time;  // Host-to-Device time
    
    cudaEventRecord(start);  // Start timer
    // Copy input arrays from CPU RAM to GPU VRAM
    cudaMemcpy(d_a, host_array_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_array_b, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);   // Stop timer
    
    cudaEventSynchronize(stop);  // Wait for GPU to finish
    cudaEventElapsedTime(&h2d_time, start, stop);  // Get elapsed time in ms
    
    // ====================================
    // STEP 7: CONFIGURE AND LAUNCH KERNEL
    // ====================================
    // Grid configuration: How many blocks and threads?
    int NB_THREAD = 256;  // Threads per block (typical: 128, 256, 512)
    int NB_BLOCK = (element + NB_THREAD - 1) / NB_THREAD;  // Ceiling division
    
    // Example: 16M elements / 256 threads = 65,536 blocks
    
    float kernel_time;
    cudaEventRecord(start);
    
    // Launch kernel with <<<blocks, threads>>> syntax
    // CRITICAL: Must pass DEVICE pointers (d_a, d_b, d_c), NOT host pointers!
    vectorAdd<<<NB_BLOCK, NB_THREAD>>>(d_a, d_b, d_c, element);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // ====================================
    // STEP 8: TRANSFER RESULTS DEVICE → HOST
    // ====================================
    // Results are in GPU memory (d_c)
    // We need to copy them back to CPU to use them
    float d2h_time;  // Device-to-Host time
    
    cudaEventRecord(start);
    cudaMemcpy(host_array_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);
    
    // ====================================
    // STEP 9: VERIFY RESULTS
    // ====================================
    // Always verify GPU computations!
    // Check if results are correct: 1.0 + 2.0 = 3.0
    bool correct = true;
    for (size_t i = 0; i < element && correct; i++) {
        if (host_array_c[i] != 3.0f) {
            printf("Error at index %zu: expected 3.0, got %.2f\n", i, host_array_c[i]);
            correct = false;
        }
    }
    if (correct) printf("Results verified!\n");
    
    // ====================================
    // STEP 10: ANALYZE PERFORMANCE
    // ====================================
    float total_time = h2d_time + kernel_time + d2h_time;
    
    // Bandwidth calculation: (data_size_GB) / (time_seconds)
    // H2D: We transfer 2 arrays (A and B)
    float h2d_bandwidth = (2 * bytes / 1e9) / (h2d_time / 1000.0);
    // D2H: We transfer 1 array (C)
    float d2h_bandwidth = (bytes / 1e9) / (d2h_time / 1000.0);
    
    printf("\n=== Performance Results ===\n");
    printf("H2D Transfer: %6.2f ms (%5.2f GB/s) - Bottleneck: PCIe bandwidth\n", 
           h2d_time, h2d_bandwidth);
    printf("Kernel:       %6.2f ms            - Actual computation on GPU\n", 
           kernel_time);
    printf("D2H Transfer: %6.2f ms (%5.2f GB/s) - Bottleneck: PCIe bandwidth\n", 
           d2h_time, d2h_bandwidth);
    printf("Total:        %6.2f ms\n", total_time);
    
    // OBSERVATION: Notice that transfer times are often longer than kernel time!
    // This is why Unified Memory can be attractive - it reduces coding complexity

    // ====================================
    // STEP 11: CLEANUP MEMORY
    // ====================================
    // ALWAYS free allocated memory to avoid memory leaks
    
    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free CPU memory
    delete[] host_array_a;
    delete[] host_array_b;
    delete[] host_array_c;
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;   
}

/*
 * ============================================================================
 * KEY CONCEPTS - EXPLICIT MEMORY MANAGEMENT
 * ============================================================================
 * 
 * 1. TWO SEPARATE MEMORY SPACES:
 *    - Host (CPU): malloc/new → host_array_*
 *    - Device (GPU): cudaMalloc → d_*
 * 
 * 2. EXPLICIT DATA MOVEMENT:
 *    - Must manually copy: CPU → GPU (cudaMemcpy H2D)
 *    - Must manually copy: GPU → CPU (cudaMemcpy D2H)
 * 
 * 3. POINTER RULES:
 *    - Host pointers CANNOT be used in kernels
 *    - Device pointers CANNOT be dereferenced on CPU
 *    - Mixing them = segmentation fault!
 * 
 * 4. PERFORMANCE CHARACTERISTICS:
 *    - Maximum control = Maximum performance (when done right)
 *    - Transfer overhead: PCIe bandwidth ~12-16 GB/s
 *    - Kernel is fast: GPU compute is powerful
 * 
 * 5. WORKFLOW:
 *    Allocate Host → Initialize → Allocate Device → Copy H2D → 
 *    Launch Kernel → Copy D2H → Verify → Free Everything
 * 
 * ============================================================================
 * NEXT STEP: UNIFIED MEMORY VERSION
 * ============================================================================
 * 
 * We will SIMPLIFY this by:
 * - Using cudaMallocManaged() instead of separate malloc/cudaMalloc
 * - Removing ALL cudaMemcpy calls
 * - Using SINGLE set of pointers accessible from both CPU and GPU
 * 
 * Trade-off: Simpler code, but potentially slower if not optimized
 * 
 * ============================================================================
 */
