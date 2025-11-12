#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define UNROLL (4)
#define BLOCK_SIZE 32

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C) {
    for (int i = si; i < si + BLOCK_SIZE; i += UNROLL * 4) {
        for (int j = sj; j < sj + BLOCK_SIZE; j++) {
            __m256d c[4];
            for (int x = 0; x < UNROLL; x++) {
                c[x] = _mm256_load_pd(C + i + x * 4 + j * n);
            }
            for (int k = sk; k < sk + BLOCK_SIZE; k++) {
                __m256d b = _mm256_broadcast_sd(B + k + j * n);
                for (int x = 0; x < UNROLL; x++) {
                    c[x] = _mm256_add_pd(
                        c[x],
                        _mm256_mul_pd(_mm256_load_pd(A + n * k + x * 4 + i), b)
                    );
                }
            }
            for (int x = 0; x < UNROLL; x++) {
                _mm256_store_pd(C + i + x * 4 + j * n, c[x]);
            }
        }
    }
}

void dgemm(int n, double *A, double *B, double *C) {
    for (int sj = 0; sj < n; sj += BLOCK_SIZE) {
        for (int si = 0; si < n; si += BLOCK_SIZE) {
            for (int sk = 0; sk < n; sk += BLOCK_SIZE) {
                do_block(n, si, sj, sk, A, B, C);
            }
        }
    }
}

int main() {
    srand(time(NULL)); // inicializa a semente de aleatoriedade

    int n = 8192; // define um único tamanho de matriz (pode alterar se quiser)

    for (int run = 1; run <= 8; run++) {

        // Aloca memória alinhada para AVX (32 bytes)
        double* A = (double*) aligned_alloc(32, n * n * sizeof(double));
        double* B = (double*) aligned_alloc(32, n * n * sizeof(double));
        double* C = (double*) aligned_alloc(32, n * n * sizeof(double));

        if (!A || !B || !C) {
            printf("Erro ao alocar memória!\n");
            return 1;
        }

        // Preenche A e B com números aleatórios, e zera C
        for (size_t i = 0; i < n * n; i++) {
            A[i] = (double) rand() / RAND_MAX;
            B[i] = (double) rand() / RAND_MAX;
            C[i] = 0.0;
        }

        // Medir tempo
        clock_t start = clock();
        dgemm(n, A, B, C);
        clock_t end = clock();

        double tempo = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Execução %d: DGEMM MULTIPLE-PROCESSORS com matriz %dx%d terminou em %.4f segundos\n",
               run, n, n, tempo);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
