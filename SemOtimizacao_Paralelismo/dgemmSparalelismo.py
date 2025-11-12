import random
import time
import sys

# ------------------------------------------------------------------------------
# Função: gerar_matriz
# Objetivo: cria uma matriz (lista de listas em Python) de dimensão
#           num_linhas × num_colunas, preenchida com números inteiros aleatórios
#           no intervalo [1, 10].
# Observação: a geração é feita linha a linha, utilizando a função randint
#             da biblioteca random.
# Complexidade temporal: O(num_linhas · num_colunas)
# ------------------------------------------------------------------------------
def gerar_matriz(num_linhas, num_colunas):
    matriz = []
    for _ in range(num_linhas):
        # Cria uma linha da matriz contendo 'num_colunas' elementos aleatórios
        linha = [random.randint(1, 10) for _ in range(num_colunas)]
        matriz.append(linha)
    return matriz


# ------------------------------------------------------------------------------
# Função: multiplicar_matrizes
# Objetivo: realiza a multiplicação matricial clássica (sem paralelismo)
#           entre duas matrizes (matriz1 × matriz2).
#
# Pré-condição: o número de colunas da matriz1 deve ser igual ao número
#               de linhas da matriz2 (dimensões compatíveis).
# Pós-condição: retorna uma nova matriz contendo o resultado da multiplicação.
#
# Fórmula matemática: 
#   Se C = A × B, então:
#       C[i][j] = Σ ( A[i][k] · B[k][j] ), k = 0..p-1
#
# Complexidade temporal: O(n³) para matrizes quadradas de ordem n.
# Complexidade espacial: O(n²) para armazenar a matriz resultado.
# ------------------------------------------------------------------------------
def multiplicar_matrizes(matriz1, matriz2):
    # Obtém dimensões de ambas as matrizes
    linhas_matriz1 = len(matriz1)
    colunas_matriz1 = len(matriz1[0])
    linhas_matriz2 = len(matriz2)
    colunas_matriz2 = len(matriz2[0])

    # Verificação de compatibilidade: condição fundamental para multiplicação
    if colunas_matriz1 != linhas_matriz2:
        raise ValueError("O número de colunas da matriz1 deve ser igual "
                         "ao número de linhas da matriz2")

    # Inicializa a matriz resultado com zeros (dimensão: linhas_matriz1 × colunas_matriz2)
    resultado = [[0 for _ in range(colunas_matriz2)] for _ in range(linhas_matriz1)]

    # Estrutura de laços triplos: varredura completa da operação matricial
    for i in range(linhas_matriz1):              # percorre as linhas da matriz1
        for j in range(colunas_matriz2):         # percorre as colunas da matriz2
            for k in range(colunas_matriz1):     # percorre os elementos comuns
                # Produto elemento a elemento e acumulação da soma
                resultado[i][j] += matriz1[i][k] * matriz2[k][j]

    return resultado


# ------------------------------------------------------------------------------
# Programa principal
# Objetivo: executar múltiplas rodadas da multiplicação de matrizes de dimensão
#           8192 × 8192, registrando o tempo de execução em arquivos .txt.
#
# Observação: matrizes 8192×8192 consomem memória em ordem de dezenas de GB e
#             execução extremamente lenta em Python puro (sem NumPy).
#             O objetivo aqui é acadêmico (didático), não prático.
# ------------------------------------------------------------------------------
for i in range(2, 8):
    # Cria um arquivo de saída distinto para cada iteração
    with open(f"matriz_python_{i}.txt", "w") as arquivo:
        # Redireciona a saída padrão (stdout) para o arquivo
        original_stdout = sys.stdout
        sys.stdout = arquivo

        # Texto isolado (não é impresso porque não há 'print')
        "Calculo da matriz 8192 x 8192"

        # Geração das matrizes A e B
        matriz_A = gerar_matriz(8192, 8192)
        matriz_B = gerar_matriz(8192, 8192)

        # Marca tempo inicial
        inicio = time.time()

        # Realiza a multiplicação matricial
        matriz_C = multiplicar_matrizes(matriz_A, matriz_B)

        # Marca tempo final
        fim = time.time()

        # Calcula e registra o tempo de execução
        print(f"Execution time for 8192 x 8192 matrix: {round(fim - inicio, 3)} seconds")

        # Restaura a saída padrão original
        sys.stdout = original_stdout