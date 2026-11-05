/*
 * Bubble Sort paralelo com MPI - modelo Divisao e Conquista (SPMD)
 *
 * Os processos formam uma arvore binaria: o filho esquerdo de r e' 2r+1
 * e o direito e' 2r+2 (o pai de r e' (r-1)/2). Todos executam o mesmo codigo.
 *
 * Uso: mpirun -np P ./bs_mpi N [modo] [delta]
 *   N     : tamanho do vetor (10000, 100000, 1000000 ...)
 *   modo  : 0 = D&C simples (padrao)  1 = D&C com balanceamento de carga
 *   delta : (modo 0) tamanho maximo de vetor que e' conquistado (ordenado)
 *           localmente. Padrao: N / numero_de_folhas, ou seja, a arvore
 *           desce ate' usar todos os P processos (P = 2^k - 1).
 *
 * Modo 0: nos internos apenas dividem e intercalam (ficam ociosos enquanto
 *         esperam os filhos); so' as folhas ordenam com bubble sort.
 * Modo 1: cada no ordena (bubble sort) uma fatia de n/s elementos, onde s e'
 *         o numero de processos da sua subarvore; o restante e' repartido
 *         entre os filhos proporcionalmente ao tamanho de suas subarvores.
 *         Assim todos os processos ordenam a mesma quantidade (N/P) e
 *         nenhum fica ocioso. Como o custo e' quadratico, isso reduz muito
 *         o tempo do caminho critico.
 *
 * Compilar: mpicc -O2 -o bs_mpi bs_mpi.c
 * Defina -DDEBUG para imprimir os vetores (apenas com N pequeno).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TAG 0

static int rank, nprocs;
static int delta;          /* limiar de conquista (modo 0) */
static double t_bs = 0.0;  /* tempo gasto ordenando neste processo */

/* ---------- Bubble sort sequencial (codigo do enunciado) ---------- */
void bs(int n, int *vetor)
{
    int c = 0, d, troca, trocou = 1;
    while (c < (n - 1) && trocou) {
        trocou = 0;
        for (d = 0; d < n - c - 1; d++)
            if (vetor[d] > vetor[d + 1]) {
                troca = vetor[d];
                vetor[d] = vetor[d + 1];
                vetor[d + 1] = troca;
                trocou = 1;
            }
        c++;
    }
}

/* bubble sort cronometrado */
static void bs_timed(int n, int *v)
{
    double t0 = MPI_Wtime();
    bs(n, v);
    t_bs += MPI_Wtime() - t0;
}

/* Intercala dois vetores ordenados a (na elementos) e b (nb elementos)
 * em out (na+nb elementos). Versao generica da rotina de intercalacao. */
static void merge(const int *a, int na, const int *b, int nb, int *out)
{
    int i = 0, j = 0, k = 0;
    while (i < na && j < nb)
        out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
}

/* numero de processos da subarvore com raiz r */
static int subtree_size(int r)
{
    if (r >= nprocs) return 0;
    return 1 + subtree_size(2 * r + 1) + subtree_size(2 * r + 2);
}

/* ---------- Modo 0: divisao e conquista simples ---------- */
static void dc_simple(int *v, int n)
{
    int left = 2 * rank + 1, right = 2 * rank + 2;

    /* conquista: vetor pequeno o bastante ou nao ha' dois filhos */
    if (n <= delta || right >= nprocs) {
        bs_timed(n, v);
        return;
    }

    /* divisao: metade para cada filho */
    int half = n / 2;
    MPI_Request req[2];
    MPI_Isend(v,        half,     MPI_INT, left,  TAG, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(v + half, n - half, MPI_INT, right, TAG, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    /* recebe as metades ordenadas e intercala */
    MPI_Recv(v,        half,     MPI_INT, left,  TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(v + half, n - half, MPI_INT, right, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int *aux = malloc(sizeof(int) * n);
    merge(v, half, v + half, n - half, aux);
    memcpy(v, aux, sizeof(int) * n);
    free(aux);
}

/* ---------- Modo 1: divisao e conquista com balanceamento ---------- */
static void dc_balanced(int *v, int n)
{
    int left = 2 * rank + 1, right = 2 * rank + 2;
    int s  = subtree_size(rank);
    int sl = subtree_size(left), sr = subtree_size(right);

    if (s == 1) {            /* folha: ordena tudo */
        bs_timed(n, v);
        return;
    }

    /* fatia propria + fatias dos filhos proporcionais as subarvores */
    int own = n / s;
    int rest = n - own;
    int nl = (int)((long)rest * sl / (s - 1));
    int nr = rest - nl;      /* nr == 0 se nao ha' filho direito */

    MPI_Request req[2];
    int nreq = 0;
    MPI_Isend(v + own, nl, MPI_INT, left, TAG, MPI_COMM_WORLD, &req[nreq++]);
    if (sr > 0)
        MPI_Isend(v + own + nl, nr, MPI_INT, right, TAG, MPI_COMM_WORLD, &req[nreq++]);

    bs_timed(own, v);        /* trabalha enquanto os filhos trabalham */

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    MPI_Recv(v + own, nl, MPI_INT, left, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (sr > 0)
        MPI_Recv(v + own + nl, nr, MPI_INT, right, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* intercalacao das tres partes ordenadas */
    int *t1 = malloc(sizeof(int) * (own + nl));
    int *t2 = malloc(sizeof(int) * n);
    merge(v, own, v + own, nl, t1);
    merge(t1, own + nl, v + own + nl, nr, t2);
    memcpy(v, t2, sizeof(int) * n);
    free(t1);
    free(t2);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int N    = (argc > 1) ? atoi(argv[1]) : 40;
    int mode = (argc > 2) ? atoi(argv[2]) : 0;
    int leaves = (nprocs + 1) / 2;
    delta = (argc > 3) ? atoi(argv[3]) : (N + leaves - 1) / leaves;

    int *vetor = NULL, n = 0;
    double t_start = 0.0, t_total = 0.0, t_proc0 = MPI_Wtime();

    if (rank == 0) {
        n = N;
        vetor = malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++)          /* pior caso para o bubble sort */
            vetor[i] = n - i;
#ifdef DEBUG
        printf("\nVetor: ");
        for (int i = 0; i < n; i++) printf("[%03d] ", vetor[i]);
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        t_start = MPI_Wtime();
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        /* recebe o vetor do pai; tamanho descoberto via Probe */
        int parent = (rank - 1) / 2;
        MPI_Status st;
        MPI_Probe(parent, TAG, MPI_COMM_WORLD, &st);
        MPI_Get_count(&st, MPI_INT, &n);
        vetor = malloc(sizeof(int) * (n > 0 ? n : 1));
        MPI_Recv(vetor, n, MPI_INT, parent, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_work0 = MPI_Wtime();
    if (mode == 0) dc_simple(vetor, n);
    else           dc_balanced(vetor, n);
    double t_work = MPI_Wtime() - t_work0;

    if (rank != 0) {   /* devolve o vetor ordenado ao pai */
        MPI_Send(vetor, n, MPI_INT, (rank - 1) / 2, TAG, MPI_COMM_WORLD);
    }

    /* estatisticas por processo para analise de balanceamento */
    double mine[2] = { t_bs, t_work }, *all = NULL;
    if (rank == 0) {
        t_total = MPI_Wtime() - t_start;
        all = malloc(sizeof(double) * 2 * nprocs);
    }
    MPI_Gather(mine, 2, MPI_DOUBLE, all, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int ok = 1;
        for (int i = 0; i < n - 1; i++)
            if (vetor[i] > vetor[i + 1]) { ok = 0; break; }
#ifdef DEBUG
        printf("\nVetor: ");
        for (int i = 0; i < n; i++) printf("[%03d] ", vetor[i]);
        printf("\n");
#endif
        printf("N=%d procs=%d modo=%d ordenado=%s tempo=%.4f s\n",
               N, nprocs, mode, ok ? "OK" : "ERRO", t_total);

        /* tempo de bubble sort (calculo) e tempo total por processo,
         * agrupados por nivel da arvore (nivel = floor(log2(rank+1))) */
        if (getenv("STATS")) {
            printf("rank nivel t_bubble t_total_no\n");
            for (int r = 0; r < nprocs; r++) {
                int lvl = 0, x = r + 1;
                while (x > 1) { x >>= 1; lvl++; }
                printf("%4d %5d %9.4f %10.4f\n", r, lvl, all[2*r], all[2*r+1]);
            }
        }
        free(all);
    }

    free(vetor);
    (void)t_proc0;
    MPI_Finalize();
    return 0;
}
