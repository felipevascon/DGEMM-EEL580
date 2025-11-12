#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define UNROLL (4)
#define REPETICOES (8)

// DGEMM opt
void dgemm(size_t n, double *A, double *B, double *C){
    for(size_t i = 0; i < n; i += UNROLL*4){

        for(size_t j = 0; j < n; j++){
            __m256d c[UNROLL];

            for(size_t x = 0; x < UNROLL; x++){
                c[x] = _mm256_load_pd(C+i+x*4+j*n);           
            }

            for(size_t k = 0; k < n; k++){
                __m256d b = _mm256_broadcast_sd(B+k+j*n);

                for(size_t x = 0; x < UNROLL; x++){
                    c[x] = _mm256_add_pd(
                        c[x],
                        _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b)
                    );
                }

                for(size_t x = 0; x < UNROLL; x++){
                    _mm256_store_pd(C+i+x*4+j*n, c[x]);
                }
            }
        }
    }
}

int main() {
    size_t n = 8192;  // tamanho da matriz

    // Alocar memória alinhada em 32 bytes
    double* A = (double*) _mm_malloc(n * n * sizeof(double), 32);
    double* B = (double*) _mm_malloc(n * n * sizeof(double), 32);
    double* C = (double*) _mm_malloc(n * n * sizeof(double), 32);

    if (!A || !B || !C) {
        printf("Erro ao alocar memória!\n");
        return 1;
    }

    // Preencher A e B com números aleatórios, C com zeros
    srand((unsigned) time(NULL));
    for (size_t i = 0; i < n * n; i++) {
        A[i] = (double) rand() / RAND_MAX;
        B[i] = (double) rand() / RAND_MAX;
        C[i] = 0.0;
    }

printf("Executando DGEMM %d vezes para matriz %zux%zu...\n\n", REPETICOES, n, n);

    for (int rep = 1; rep <= REPETICOES; rep++) {

        // Reinicializa C com zero a cada rodada
        for (size_t i = 0; i < n * n; i++) {
            C[i] = 0.0;
        }

        clock_t start = clock();
        dgemm(n, A, B, C);
        clock_t end = clock();

        double tempo = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Execução %d: %.4f segundos\n", rep, tempo);
    }

    // Liberar memória
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
