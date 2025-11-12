#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

#define UNROLL (4)
#define BLOCKSIZE 32

void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C){

    for(int i = si; i < si+BLOCKSIZE; i+= UNROLL*4){
        for(int j = sj; j < sj+BLOCKSIZE; j++){
            __m256d c[4];
            for(int x = 0; x < UNROLL; x++){
                c[x] = _mm256_load_pd(C+i+x*4+j*n);
            }
            for(int k = sk; k < sk+BLOCKSIZE; k++){
                __m256d b = _mm256_broadcast_sd(B+k+j*n);
                for(int x = 0; x < UNROLL; x++){
                    c[x] = _mm256_add_pd(
                        c[x],
                        _mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b)
                    );
                }
            }
            for(int x = 0; x < UNROLL; x++){
                _mm256_store_pd(C+i+x*4+j*n, c[x]);
            }
        }
    }
}

// DGEMM opt
void dgemm(size_t n, double *A, double *B, double *C){
    
    for(int sj = 0; sj < n; sj += BLOCKSIZE){
        for(int si = 0; si < n; si += BLOCKSIZE){
            for(int sk = 0; sk < n; sk += BLOCKSIZE){
                do_block(n, si, sj, sk, A, B, C);
            }
        }
    }
}

int main() {
    srand(time(NULL)); // inicializa a semente
    int sizes[] = {256, 512, 1024}; // Array com o tam das matrizes
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]); // Num de elem do array (no caso 6)
    
    /*  sizeof(sizes) retorna "num elem do array" x "tam tipo int (= 4 bytes)"
        sizeof(sizes) nesse caso = 6 x 4
        sizeof(sizes[0]) retorna "sizeof(int)" = 4 bytes
        num_sizes nesse caso faz 6x4/ 4 = 24/4 = 6 */

    // executa 8x pra automatizar os testes
    for(int run = 1; run <= 8; run++){

        /*  define o nome do arq de saida (monta o nome dinamicamente)
            ex: run = 1 --> "dgemm_1.txt"
                run = 2 --> "dgemm_2.txt" */
        char filename[40];
        snprintf(filename, sizeof(filename), "dgemm_cache_b_%d.txt", run);

        printf("Execucao %d:\n", run);

        // abre o arquivo em modo de escrita
        FILE *file = fopen(filename, "w");
        if(file == NULL){
            perror("Erro ao abrir o arquivo p/ escrita");
            return 1;
        }

        for(int i = 0; i < num_sizes; i++){
            int n = sizes[i]; // tamanho da matriz

            // Alocar memória para as matrizes (vetor linear de tamanho n*n)
            double* A = (double*) malloc(n * n * sizeof(double));
            double* B = (double*) malloc(n * n * sizeof(double));
            double* C = (double*) malloc(n * n * sizeof(double));
     
            if(!A || !B || !C){
                printf("Erro ao alocar memória!\n");
                return 1;
            }

            // Preencher A e B com números aleatórios, C com zeros
            for(size_t i = 0; i < n * n; i++){
                A[i] = (double) rand() / RAND_MAX;
                B[i] = (double) rand() / RAND_MAX;
                C[i] = 0.0;
            }

            // Medir tempo
            clock_t start = clock(); // tempo de inicio
            dgemm(n, A, B, C); // multiplica A e B e guarda em C
            clock_t end = clock(); // tempo de termino

            // calcula o tempo decorrido em seg
            double tempo = (double)(end - start) / CLOCKS_PER_SEC;
            printf("DGEMM CACHE BLOCKING com matriz %zux%zu terminou em %.4f segundos\n", n, n, tempo);

            // grava no arquivo
            fprintf(file, "DGEMM CACHE BLOCKING com matriz %zux%zu terminou em %.4f segundos\n", n, n, tempo);
            
            // Liberar memória
            free(A);
            free(B);
            free(C);
        }

        // fecha o arquivo
        fclose(file);
        printf("\n");
    }

    return 0;
}
