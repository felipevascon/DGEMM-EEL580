#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#define UNROLL (4)
#define BLOCK_SIZE 32
#define REPETICOES (8)
void do_block(int n, int si, int sj, int sk,double *A, double *B, double *C){
    
    for (int i = si; i < si + BLOCK_SIZE; i += UNROLL * 4)
        for (int j = sj; j < sj + BLOCK_SIZE; ++j){
            __m256d c[UNROLL];
            for (int r = 0; r < UNROLL; r++)
                c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
            for (int k = sk; k < sk + BLOCK_SIZE; k++){
                __m256d bb = _mm256_broadcast_sd(B + j * n + k);
            for (int r = 0; r < UNROLL; r++) {
            // Load elements from arrays A and perform manual multiply and add
            __m256d a = _mm256_load_pd(A + n * k + r * 4 + i);
            c[r] = _mm256_add_pd(c[r], _mm256_mul_pd(a, bb));
        }
    }
            for (int r = 0; r < UNROLL; r++)
                _mm256_store_pd(C + i + r * 4 + j * n, c[r]);
        }
        }
void dgemm(int n, double *A, double *B, double *C) {
#pragma omp parallel for
for (int sj = 0; sj < n; sj += BLOCK_SIZE)
for (int si = 0; si < n; si += BLOCK_SIZE)
for (int sk = 0; sk < n; sk += BLOCK_SIZE)
do_block(n, si, sj, sk, A, B, C);
}
void generateRandomMatrix(int rows, int cols, double *matrix) {
for (int i = 0; i < rows * cols; i++) {
// Gerar um número inteiro aleatório entre 0 e RAND_MAX
int randomInt = rand();
// Converter o número inteiro em um valor double no intervalo [0, 1]
double randomDouble = (double)randomInt / RAND_MAX;
// Escalar o valor para o intervalo [0, 10]
matrix[i] = randomDouble * 10.0;
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
