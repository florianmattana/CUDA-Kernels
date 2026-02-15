#include"cuda_intellisense_fix.h"
#include"kernels.h"
#include<cuda_runtime.h>

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadId_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadId_y < M && threadId_x < N)
    {
        float total = 0.0f;

        for (int i = 0; i < K; ++i)
        {
            total += A[threadId_y * K + i] * B[i * N + threadId_x];
        }
        C[threadId_y * N + threadId_x] = total;
    }
};

__global__ void tiled_gemm(float* A, float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockIdx.x * BN + threadIdx.x;
    int threadId_y = blockIdx.y * BM + threadIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float total = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK)
    {

        int a_row = threadId_y;
        int a_col = k0 + threadIdx.x;

        int b_col = threadId_x;
        int b_row = k0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            total += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (threadId_y < M && threadId_x < N)
    {
        C[threadId_y * N + threadId_x] = total;
    }
};

template<int TM>
__global__ void tiled_gemm_upgrd(const float* A, const float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockIdx.x * BN + threadIdx.x;
    int threadId_y_rowBase = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM] = { 0.0f };

    int num_threads_in_block = blockDim.x * blockDim.y;
    int linear_id = threadIdx.y * blockDim.x + threadIdx.x;

    // --- Chargement collaboratif de la tuile de A en shared memory ---
    // La tuile fait BM × BK éléments mais on a moins de threads que d'éléments.
    // Chaque thread charge donc plusieurs éléments en itérant avec un stride
    // égal au nombre de threads dans le bloc.
    for (int k0 = 0; k0 < K; k0 += BK)
    {
        for (int i = linear_id; i < BM * BK; i += num_threads_in_block)
        {
            // Conversion linéaire → 2D : on retrouve la position dans la tuile
            // pour savoir OÙ ÉCRIRE dans As
            int r = i / BK;
            int c = i % BK;

            // Conversion locale → globale : on retrouve la position dans la matrice A
            // pour savoir OÙ LIRE en mémoire globale
            int gr = blockIdx.y * BM + r;
            int gc = k0 + c;

            // Transfert global → shared, avec vérification des bornes
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        for (int i = linear_id; i < BK * BN; i += num_threads_in_block)
        {
            int r = i / BN;
            int c = i % BN;
            int gr = k0 + r;
            int gc = blockIdx.x * BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // --- Produit scalaire partiel depuis la shared memory ---
        // On parcourt les BK colonnes de la tuile. Pour chaque colonne kk,
        // on charge UNE SEULE FOIS la valeur de B dans un registre,
        // puis on la réutilise TM fois : une fois par ligne gérée par ce thread.
        // C'est le gain principal de TM : une lecture, TM utilisations.
#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            // Lecture unique de B pour cette colonne kk
            // Stockée en registre (~0 cycle pour les réutilisations)
            float b = Bs[kk][threadIdx.x];

#pragma unroll
            // On boucle sur les TM lignes dont ce thread est responsable
            // threadIdx.y * TM + m donne la ligne locale dans As
            for (int m = 0; m < TM; ++m)
            {
                // acc[m] accumule le produit scalaire pour la m-ième ligne
                // La même valeur b est réutilisée pour chaque ligne
                acc[m] += As[threadIdx.y * TM + m][kk] * b;
            }
        }

        // On attend que tous les threads aient fini de lire la shared memory
        // avant que l'itération suivante de k0 ne l'écrase
        __syncthreads();
    }

    // --- Écriture des résultats dans C en mémoire globale ---
    // Chaque thread écrit ses TM résultats dans C
    // La colonne est la même pour tous, on la vérifie une seule fois
    if (threadId_x < N)
    {
#pragma unroll
        for (int m = 0; m < TM; ++m)
        {
            // La ligne globale de chaque résultat
            int r = threadId_y_rowBase + m;
            // Vérification individuelle : si TM ne divise pas M,
            // la dernière ligne d'un thread pourrait dépasser
            if (r < M)
                C[r * N + threadId_x] = acc[m];
        }
    }
}

// Instanciation explicite du template pour TM = 2
// Sans cette ligne, le compilateur ne sait pas quelle version
// du kernel générer si le lancement est dans un autre fichier
template __global__ void tiled_gemm_upgrd<2>(const float*, const float*, float*, int, int, int);