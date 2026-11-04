/*
 * Bubble Sort paralelo com MPI - modelo Coordenador/Trabalhador
 *
 * Esta e a versao com distribuicao dinamica (balanceamento de carga),
 * inspirada no codigo de Sudoku fornecido.
 *
 * Rank 0: coordenador; distribui blocos e recebe os blocos ordenados.
 * Ranks 1..P-1: trabalhadores; ordenam os blocos com Bubble Sort.
 * Ao final, o coordenador intercala os blocos ordenados.
 *
 * Para executar a versao sequencial, use mpirun -np 1.
 * Para executar a versao paralela, use mpirun -np P, com P >= 2.
 *
 * Compilacao:
 *   mpicc -O2 -Wall -Wextra -o bs_mpi bs_mpi.c
 *
 * Exemplos:
 *   mpirun -np 1 ./bs_mpi -n 10000
 *   mpirun -np 3 ./bs_mpi -n 10000 -c 1000
 *   mpirun -np 7 ./bs_mpi -n 10000 -c 400
 *   mpirun -np 15 ./bs_mpi -n 100000 -c 2000
 *
 * Opcoes:
 *   -n N   quantidade de elementos (padrao: 10000)
 *   -c C   quantidade de elementos por tarefa/bloco (padrao: calculada)
 *   -w     cria vetor em ordem decrescente (pior caso para Bubble Sort)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#define TAG_PEDIDO   1
#define TAG_TAREFA   2
#define TAG_DADOS    3
#define TAG_RESULTADO 4
#define TAG_FIM      5
#define COORDENADOR  0

/* Bubble Sort sequencial aplicado a um bloco do vetor. */
static void bs(int n, int *vetor)
{
    int c = 0;
    int trocou = 1;

    while (c < n - 1 && trocou) {
        int d;
        int troca;
        trocou = 0;

        for (d = 0; d < n - c - 1; d++) {
            if (vetor[d] > vetor[d + 1]) {
                troca = vetor[d];
                vetor[d] = vetor[d + 1];
                vetor[d + 1] = troca;
                trocou = 1;
            }
        }
        c++;
    }
}

/* Intercala dois trechos ja ordenados: [inicio, meio) e [meio, fim). */
static void intercalar(const int *origem, int *destino,
                       int inicio, int meio, int fim)
{
    int i = inicio;
    int j = meio;
    int k = inicio;

    while (i < meio && j < fim) {
        if (origem[i] <= origem[j])
            destino[k++] = origem[i++];
        else
            destino[k++] = origem[j++];
    }

    while (i < meio)
        destino[k++] = origem[i++];

    while (j < fim)
        destino[k++] = origem[j++];
}

/*
 * Intercala progressivamente os blocos ordenados.
 * Na primeira rodada, cada trecho tem ate 'tamanho_bloco' elementos.
 * Depois, os trechos intercalados dobram de tamanho a cada rodada.
 */
static void intercalar_todos(int *vetor, int n, int tamanho_bloco)
{
    int *auxiliar = malloc((size_t)n * sizeof(int));
    int *origem = vetor;
    int *destino = auxiliar;
    long largura = tamanho_bloco;

    if (!auxiliar) {
        fprintf(stderr, "Erro: memoria insuficiente na intercalacao.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    while (largura < n) {
        long passo = 2 * largura;

        for (long inicio = 0; inicio < n; inicio += passo) {
            long meio_l = inicio + largura;
            long fim_l = inicio + passo;

            if (meio_l > n) meio_l = n;
            if (fim_l > n) fim_l = n;

            intercalar(origem, destino,
                       (int)inicio, (int)meio_l, (int)fim_l);
        }

        {
            int *tmp = origem;
            origem = destino;
            destino = tmp;
        }
        largura *= 2;
    }

    if (origem != vetor)
        memcpy(vetor, origem, (size_t)n * sizeof(int));

    free(auxiliar);
}

/* Gerador deterministico para que as execucoes usem a mesma entrada. */
static uint32_t estado = 123456789u;

static uint32_t proximo_aleatorio(void)
{
    estado ^= estado << 13;
    estado ^= estado >> 17;
    estado ^= estado << 5;
    return estado;
}

static void inicializar_vetor(int *vetor, int n, int pior_caso)
{
    if (pior_caso) {
        for (int i = 0; i < n; i++)
            vetor[i] = n - i;
    } else {
        estado = 123456789u;
        for (int i = 0; i < n; i++)
            vetor[i] = (int)(proximo_aleatorio() & 0x7fffffff);
    }
}

static int validar_vetor(const int *vetor, int n)
{
    for (int i = 1; i < n; i++) {
        if (vetor[i - 1] > vetor[i])
            return 0;
    }
    return 1;
}

static uint64_t checksum(const int *vetor, int n)
{
    uint64_t h = 1469598103934665603ULL;

    for (int i = 0; i < n; i++) {
        h ^= (uint32_t)vetor[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/*
 * Coordenador: entrega blocos a quem estiver disponivel.
 * O rank 0 nao executa Bubble Sort nesta implementacao.
 */
static void coordenador(int *vetor, int n, int tamanho_bloco, int np)
{
    int trabalhadores = np - 1;
    int proximo = 0;
    int encerrados = 0;

    int *inicio_por_rank = calloc((size_t)np, sizeof(int));
    int *qtd_por_rank = calloc((size_t)np, sizeof(int));

    if (!inicio_por_rank || !qtd_por_rank) {
        fprintf(stderr, "Erro: memoria insuficiente no coordenador.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    while (encerrados < trabalhadores) {
        MPI_Status status;
        int origem;
        int tag;

        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        origem = status.MPI_SOURCE;
        tag = status.MPI_TAG;

        if (tag == TAG_RESULTADO) {
            int qtd = qtd_por_rank[origem];

            MPI_Recv(vetor + inicio_por_rank[origem], qtd, MPI_INT,
                     origem, TAG_RESULTADO, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else if (tag == TAG_PEDIDO) {
            MPI_Recv(NULL, 0, MPI_BYTE, origem, TAG_PEDIDO,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            fprintf(stderr, "Coordenador recebeu uma mensagem inesperada.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /*
         * Tanto um pedido inicial quanto a devolucao de um resultado
         * deixam o trabalhador disponivel para receber outra tarefa.
         */
        if (proximo < n) {
            int cabecalho[2];
            int qtd = n - proximo;

            if (qtd > tamanho_bloco)
                qtd = tamanho_bloco;

            cabecalho[0] = proximo;
            cabecalho[1] = qtd;

            inicio_por_rank[origem] = proximo;
            qtd_por_rank[origem] = qtd;

            MPI_Send(cabecalho, 2, MPI_INT, origem, TAG_TAREFA,
                     MPI_COMM_WORLD);
            MPI_Send(vetor + proximo, qtd, MPI_INT, origem, TAG_DADOS,
                     MPI_COMM_WORLD);

            proximo += qtd;
        } else {
            MPI_Send(NULL, 0, MPI_BYTE, origem, TAG_FIM, MPI_COMM_WORLD);
            encerrados++;
        }
    }

    free(inicio_por_rank);
    free(qtd_por_rank);

    /* Neste ponto, todos os blocos estao ordenados em suas posicoes. */
    intercalar_todos(vetor, n, tamanho_bloco);
}

/* Trabalhador: solicita blocos, ordena cada um e devolve o resultado. */
static void trabalhador(int tamanho_bloco)
{
    int *bloco = malloc((size_t)tamanho_bloco * sizeof(int));

    if (!bloco) {
        fprintf(stderr, "Erro: trabalhador sem memoria.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* O primeiro pedido inicia o atendimento deste trabalhador. */
    MPI_Send(NULL, 0, MPI_BYTE, COORDENADOR, TAG_PEDIDO,
             MPI_COMM_WORLD);

    for (;;) {
        int cabecalho[2];
        MPI_Status status;

        /*
         * Apos cada resultado, o coordenador responde diretamente
         * com outra tarefa ou com TAG_FIM.
         */
        MPI_Recv(cabecalho, 2, MPI_INT, COORDENADOR, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_FIM)
            break;

        if (status.MPI_TAG != TAG_TAREFA) {
            fprintf(stderr, "Trabalhador recebeu uma mensagem inesperada.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        {
            int quantidade = cabecalho[1];

            MPI_Recv(bloco, quantidade, MPI_INT, COORDENADOR, TAG_DADOS,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            bs(quantidade, bloco);

            /* A devolucao do resultado tambem sinaliza disponibilidade. */
            MPI_Send(bloco, quantidade, MPI_INT, COORDENADOR,
                     TAG_RESULTADO, MPI_COMM_WORLD);
        }
    }

    free(bloco);
}

int main(int argc, char **argv)
{
    int rank, np;
    int n = 10000;
    int tamanho_bloco = 0;
    int pior_caso = 0;
    int argumentos_ok = 1;
    int *vetor = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-c") && i + 1 < argc) {
            tamanho_bloco = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-w")) {
            pior_caso = 1;
        } else {
            argumentos_ok = 0;
        }
    }

    if (!argumentos_ok || n < 1 || tamanho_bloco < 0) {
        if (rank == COORDENADOR)
            fprintf(stderr,
                    "Uso: mpirun -np P %s [-n N] [-c TAM_BLOCO] [-w]\n",
                    argv[0]);
        MPI_Finalize();
        return 1;
    }

    /*
     * Se o tamanho do bloco nao foi informado, cria aproximadamente
     * quatro tarefas por trabalhador, favorecendo a distribuicao dinamica.
     */
    if (tamanho_bloco == 0) {
        int trabalhadores = (np > 1) ? np - 1 : 1;
        tamanho_bloco = n / (4 * trabalhadores);
        if (tamanho_bloco < 1)
            tamanho_bloco = 1;
    }

    if (rank == COORDENADOR) {
        vetor = malloc((size_t)n * sizeof(int));
        if (!vetor) {
            fprintf(stderr, "Erro: memoria insuficiente para o vetor.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        inicializar_vetor(vetor, n, pior_caso);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double inicio = MPI_Wtime();

    if (np == 1) {
        /* Execucao sequencial de referencia. */
        if (rank == COORDENADOR)
            bs(n, vetor);
    } else {
        if (rank == COORDENADOR)
            coordenador(vetor, n, tamanho_bloco, np);
        else
            trabalhador(tamanho_bloco);
    }

    double tempo = MPI_Wtime() - inicio;

    if (rank == COORDENADOR) {
        int ordenado = validar_vetor(vetor, n);

        printf("Modo: %s\n", np == 1 ? "sequencial" : "MPI coordenador/trabalhador");
        printf("Elementos: %d\n", n);
        printf("Processos MPI: %d\n", np);
        if (np > 1)
            printf("Trabalhadores: %d\nBloco: %d elementos\n",
                   np - 1, tamanho_bloco);
        printf("Tempo: %.6f segundos\n", tempo);
        printf("Vetor ordenado: %s\n", ordenado ? "SIM" : "NAO");
        printf("Checksum: %016llx\n",
               (unsigned long long)checksum(vetor, n));

        if (!ordenado)
            fprintf(stderr, "ERRO: o vetor final nao esta ordenado.\n");

        free(vetor);
    }

    MPI_Finalize();
    return 0;
}
