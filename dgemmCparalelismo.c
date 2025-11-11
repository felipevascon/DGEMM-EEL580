#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

// DGEMM simples
void dgemm(size_t n, double* A, double* B, double* C){
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j) {
            double cij = C[i + j*n];   // C[i][j]
            for(size_t k = 0; k < n; ++k)
                cij += A[i + k*n] * B[k + j*n];
            C[i + j*n] = cij;
        }
}

int main(){

    srand(time(NULL)); // inicializa a semente
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192}; // Array com o tam das matrizes
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
        snprintf(filename, sizeof(filename), "dgemm_%d.txt", run);

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
            printf("DGEMM com matriz %zux%zu terminou em %.4f segundos\n", n, n, tempo);

            // grava no arquivo
            fprintf(file, "DGEMM com matriz %zux%zu terminou em %.4f segundos\n", n, n, tempo);
            
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
