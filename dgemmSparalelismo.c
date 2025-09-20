#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 Função para multiplicar duas matrizes quadradas A e B,
 armazenando o resultado em C.
 Implementação direta sem paralelismo, usando dgemm clássico, sem otimizações de BLAS.
*/
void dgemm(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i++) {                               // Percorre as linhas de A
        for (int j = 0; j < n; j++) {                           // Percorre as colunas de B
            C[i * n + j] = 0.0;                                 // Inicializa elemento C[i][j]
            for (int k = 0; k < n; k++) {                       // Produto escalar da linha i (A) e coluna j (B)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

/*
 Função para gerar uma matriz de dimensão (rows x cols)
 preenchida com valores aleatórios no intervalo [0, 10].
*/
void generateRandomMatrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows * cols; i++) {                     // Percorre cada elemento da matriz p gerar:  
        int randomInt = rand();                                 // Inteiro aleatório entre 0 e RAND_MAX
        double randomDouble = (double)randomInt / RAND_MAX;     // Normaliza para [0,1] (convertendo int p/ double)
        matrix[i] = randomDouble * 10.0;                        // Escalar o valor para [0,10]
    }
}

int main() {
    // Inicializa a semente do gerador de números aleatórios com o tempo atual
    srand(time(NULL));

    // Conjunto de tamanhos de matrizes a serem testados
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Executa 3 rodadas (run = 6, 7 e 8)
    for (int run = 6; run <= 8; run++) {
        char filename[20];
        snprintf(filename, sizeof(filename), "matrizes_c_%d.txt", run);

        printf("Execução %d:\n", run);

        // Abre o arquivo de saída para salvar tempos de execução
        FILE *file = fopen(filename, "w");
        if (file == NULL) {
            perror("Erro ao abrir o arquivo para escrita");
            return 1;
        }

        // Testa cada tamanho definido em 'sizes'
        for (int i = 0; i < num_sizes; i++) {
            int n = sizes[i];   // Tamanho da matriz (n x n)

            // Aloca dinamicamente memória para A, B e C
            double *A = (double *)malloc(n * n * sizeof(double));
            double *B = (double *)malloc(n * n * sizeof(double));
            double *C = (double *)malloc(n * n * sizeof(double));

            // Preenche A e B com valores aleatórios
            generateRandomMatrix(n, n, A);
            generateRandomMatrix(n, n, B);

            // Marca tempo inicial
            clock_t start = clock();

            // Realiza a multiplicação A * B → C
            dgemm(n, A, B, C);

            // Marca tempo final
            clock_t end = clock();

            // Calcula tempo decorrido em segundos
            double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

            // Exibe no console
            printf("Matrix size: %dx%d, Execution time: %.3lf seconds\n",
                   n, n, elapsed_time);

            // Grava o tempo no arquivo
            fprintf(file, "Matrix size: %dx%d, Execution time: %.3lf seconds\n",
                    n, n, elapsed_time);

            // Libera a memória alocada para as matrizes
            free(A);
            free(B);
            free(C);
        }

        // Fecha o arquivo
        fclose(file);
        printf("Arquivo '%s' salvo com sucesso!\n", filename);
        printf("\n");
    }

    return 0;
}