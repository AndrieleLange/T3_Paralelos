/*
  PARALELO COM MPI - modelo COORDENADOR / TRABALHADOR (saco de trabalho)

  O solver (sd_genmat, sd_update, sd_solve) e IDENTICO ao de sudoku_seq.c. O carregamento do lote tambem, para N <= tamanho do arquivo (lote completo e escalabilidade forte, base da fraca); acima disso este arquivo replica o arquivo inteiro em vez de esticar (ver carrega_lote). O resto que muda e o driver (main).

  SACO DE TRABALHO: o lote de N puzzles e uma matriz N x 82 (cada linha e um "vetor" de 81 caracteres + '\0'). A unidade de trabalho e um bloco de U linhas consecutivas (U = -u, padrao 16). So o coordenador le o arquivo e guarda o saco.

  PROTOCOLO (a iniciativa e sempre do trabalhador):
    trabalhador -> coordenador  TAG_PEDIDO     (vazia) primeiro pedido de trabalho
    coordenador -> trabalhador  TAG_TRABALHO   bloco de ate U puzzles
    trabalhador -> coordenador  TAG_RESULTADO  bloco de solucoes; vale tambem como novo pedido
    coordenador -> trabalhador  TAG_FIM        (vazia) saco vazio, pode terminar

  O coordenador recebe com MPI_ANY_SOURCE, ou seja, atende quem ficar pronto primeiro. Para manter a ordem original ele anota qual bloco cada trabalhador tem em maos e recebe o resultado DIRETO na posicao correta da matriz de solucoes; por isso a mensagem de resultado nao precisa carregar o indice.

  O programa e SPMD: o mesmo binario roda em todos os processos, o rank 0 e o coordenador e os ranks 1..np-1 sao trabalhadores. Funciona para qualquer np >= 2.
*/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>


/* Representacao esparsa da matriz binaria 729x324 de cobertura exata. */
typedef struct {
	uint16_t r[324][9];
	uint16_t c[729][4];
} sdaux_t;


/* Gera a matriz de restricoes. Chamada uma unica vez. */
static sdaux_t *sd_genmat(void)
{
	sdaux_t *a;
	int i, j, k, r, c, c2;
	int8_t nr[324];
	a = calloc(1, sizeof(sdaux_t));
	if (!a) { fprintf(stderr, "erro: sem memoria\n"); exit(1); }
	for (i = r = 0; i < 9; ++i)
		for (j = 0; j < 9; ++j)
			for (k = 0; k < 9; ++k)
				a->c[r][0] = 9 * i + j,                  /* linha-coluna  */
				a->c[r][1] = (i/3*3 + j/3) * 9 + k + 81, /* bloco-numero  */
				a->c[r][2] = 9 * i + k + 162,            /* linha-numero  */
				a->c[r][3] = 9 * j + k + 243,            /* coluna-numero */
				++r;
	for (c = 0; c < 324; ++c) nr[c] = 0;
	for (r = 0; r < 729; ++r)
		for (c2 = 0; c2 < 4; ++c2)
			k = a->c[r][c2], a->r[k][nr[k]++] = r;
	return a;
}

/* Aplica (v=+1) ou desfaz (v=-1) a escolha r, mantendo os contadores de escolhas disponiveis sincronizados. Retorna a restricao mais restrita. */
static inline int sd_update(const sdaux_t *aux, int8_t sr[729], uint8_t sc[324], int r, int v)
{
	int c2, min = 10, min_c = 0;
	for (c2 = 0; c2 < 4; ++c2) sc[aux->c[r][c2]] += v<<7;
	for (c2 = 0; c2 < 4; ++c2) {
		int r2, rr, cc2, c = aux->c[r][c2];
		if (v > 0) {
			for (r2 = 0; r2 < 9; ++r2) {
				if (sr[rr = aux->r[c][r2]]++ != 0) continue;
				for (cc2 = 0; cc2 < 4; ++cc2) {
					int cc = aux->c[rr][cc2];
					if (--sc[cc] < min) min = sc[cc], min_c = cc;
				}
			}
		} else {
			const uint16_t *p;
			for (r2 = 0; r2 < 9; ++r2) {
				if (--sr[rr = aux->r[c][r2]] != 0) continue;
				p = aux->c[rr]; ++sc[p[0]]; ++sc[p[1]]; ++sc[p[2]]; ++sc[p[3]];
			}
		}
	}
	return min<<16 | min_c;
}

/* Resolve um Sudoku; devolve o numero de solucoes e copia a primeira para 'solucao' (82 bytes). Todo o estado da busca fica na pilha, sem variaveis globais: cada puzzle e independente dos outros, o que permite resolve-los em qualquer trabalhador e em qualquer ordem. */
static int sd_solve(const sdaux_t *aux, const char *_s, char *solucao)
{
	int i, j, r, c, r2, dir, cand, n = 0, min, hints = 0;
	int8_t sr[729], cr[81];
	uint8_t sc[324];
	int16_t cc[81];
	char out[82];

	for (r = 0; r < 729; ++r) sr[r] = 0;
	for (c = 0; c < 324; ++c) sc[c] = 0<<7|9;
	for (i = 0; i < 81; ++i) {
		int a = _s[i] >= '1' && _s[i] <= '9' ? _s[i] - '1' : -1;
		if (a >= 0) sd_update(aux, sr, sc, i * 9 + a, 1);
		if (a >= 0) ++hints;
		cr[i] = cc[i] = -1, out[i] = _s[i];
	}
	for (i = 0, dir = 1, cand = 10<<16|0, out[81] = 0;;) {
		while (i >= 0 && i < 81 - hints) {
			if (dir == 1) {
				min = cand>>16, cc[i] = cand&0xffff;
				if (min > 1) {
					for (c = 0; c < 324; ++c) {
						if (sc[c] < min) {
							min = sc[c], cc[i] = c;
							if (min <= 1) break;
						}
					}
				}
				if (min == 0 || min == 10) cr[i--] = dir = -1;
			}
			c = cc[i];
			if (dir == -1 && cr[i] >= 0) sd_update(aux, sr, sc, aux->r[c][cr[i]], -1);
			for (r2 = cr[i] + 1; r2 < 9; ++r2)
				if (sr[aux->r[c][r2]] == 0) break;
			if (r2 < 9) {
				cand = sd_update(aux, sr, sc, aux->r[c][r2], 1);
				cr[i++] = r2; dir = 1;
			} else cr[i--] = dir = -1;
		}
		if (i < 0) break;
		for (j = 0; j < i; ++j) r = aux->r[cc[j]][cr[j]], out[r/9] = r%9 + '1';
		if (n == 0 && solucao) memcpy(solucao, out, 82);
		++n; --i; dir = -1;
	}
	return n;
}

/* --------------------------------------------------------------------------
 * DRIVER DE LOTE (nosso)
 * ------------------------------------------------------------------------ */
#define LARGURA 82  /* 81 caracteres + terminador */

/* Checksum FNV-1a das solucoes, na ordem do indice do puzzle. Como nao depende da ordem de execucao, sequencial e paralelo devem dar o mesmo valor, e o teste de regressao contra condicoes de corrida. */
static uint64_t checksum(const char *solucoes, long n)
{
	uint64_t h = 1469598103934665603ULL;
	for (long i = 0; i < n * LARGURA; ++i) {
		h ^= (unsigned char)solucoes[i];
		h *= 1099511628211ULL;
	}
	return h;
}

/* PRNG proprio (splitmix64) em vez de rand(): a libc varia entre plataformas e o mesmo seed daria ordens diferentes no Linux e no macOS. Assim o embaralhamento e reprodutivel bit a bit em qualquer maquina. */
static uint64_t estado_prng;
static uint64_t proximo_aleatorio(void)
{
	uint64_t z = (estado_prng += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

/* Embaralha o lote em memoria com seed fixo. O arquivo puzzles.txt esta ORDENADO POR DIFICULDADE: os puzzles baratos no inicio, os caros no fim. Essa e a situacao ruim para a distribuicao em bloco estatico (um unico bloco contiguo por trabalhador, -u = N/W), que
 * entrega quase todo o trabalho pesado ao ultimo bloco. Embaralhar dispersa os custos e permite comparar os dois cenarios usando o MESMO conjunto de puzzles (mesmo trabalho total). */
static void embaralha(char *lote, long n, uint64_t seed)
{
	estado_prng = seed;
	for (long i = n - 1; i > 0; --i) {
		uint64_t limite = (uint64_t)i + 1;
		uint64_t zona = UINT64_MAX - (UINT64_MAX % limite), r;
		do { r = proximo_aleatorio(); } while (r >= zona);
		long j = (long)(r % limite);
		char tmp[LARGURA];
		memcpy(tmp, lote + i * LARGURA, LARGURA);
		memcpy(lote + i * LARGURA, lote + j * LARGURA, LARGURA);
		memcpy(lote + j * LARGURA, tmp, LARGURA);
	}
}

/* Carrega o arquivo e ajusta o lote para 'desejados' puzzles (necessario para a escalabilidade fraca).
 * - desejados < arquivo: amostra espalhada pelo arquivo todo, preservando a distribuicao de dificuldade (igual ao sudoku_seq.c, mesmo checksum).
 * - desejados > arquivo: copias inteiras do arquivo, uma apos a outra, e a sobra amostrada espalhada.
 *   DIFERENTE do sudoku_seq.c, que "estica" o arquivo (cada puzzle aparece varias vezes SEGUIDAS). Resolver o mesmo puzzle repetidamente em sequencia fica ~20% mais rapido (cache e preditor de desvios aquecidos), o que inflaria a eficiencia fraca, cuja base T_seq(6000) nao tem repeticao. Com copias inteiras o custo medio por puzzle fica constante em qualquer N. */
static char *carrega_lote(const char *caminho, long desejados, long *n_final, long *n_unicos)
{
	FILE *f = fopen(caminho, "r");
	if (!f) { perror(caminho); exit(1); }

	long capacidade = 65536, n = 0;
	char *lote = malloc((size_t)capacidade * LARGURA);
	if (!lote) { fprintf(stderr, "erro: sem memoria\n"); exit(1); }

	char linha[1024];
	while (fgets(linha, sizeof linha, f)) {
		if (linha[0] == '#' || strlen(linha) < 81) continue;
		if (n == capacidade) {
			capacidade *= 2;
			lote = realloc(lote, (size_t)capacidade * LARGURA);
			if (!lote) { fprintf(stderr, "erro: sem memoria\n"); exit(1); }
		}
		memcpy(lote + n * LARGURA, linha, 81);
		lote[n * LARGURA + 81] = '\0';
		++n;
	}
	fclose(f);
	if (n == 0) { fprintf(stderr, "erro: nenhum puzzle valido em %s\n", caminho); exit(1); }

	*n_unicos = n;
	if (desejados > 0 && desejados != n) {
		char *novo = malloc((size_t)desejados * LARGURA);
		if (!novo) { fprintf(stderr, "erro: sem memoria\n"); exit(1); }
		long copias = desejados / n, resto = desejados % n;
		for (long i = 0; i < desejados; ++i) {
			long j = (i < copias * n) ? i % n                        /* copias inteiras do arquivo */
			                          : (i - copias * n) * n / resto; /* sobra: amostra espalhada */
			memcpy(novo + i * LARGURA, lote + j * LARGURA, LARGURA);
		}
		free(lote);
		lote = novo;
		n = desejados;
	}
	*n_final = n;
	return lote;
}

/* --------------------------------------------------------------------------
 * COORDENADOR / TRABALHADOR (nosso)
 * ------------------------------------------------------------------------ */
#define COORDENADOR   0
#define TAG_PEDIDO    1
#define TAG_TRABALHO  2
#define TAG_RESULTADO 3
#define TAG_FIM       4

/* Contabilidade de carga de cada trabalhador numa rodada, usada para a analise de balanceamento. */
typedef struct {
	double tempo_calculo;  /* tempo dentro do sd_solve */
	double tempo_total;    /* do primeiro pedido ate receber TAG_FIM */
	double unidades;       /* blocos processados (double so para caber num MPI_Gather de doubles) */
	double puzzles;        /* puzzles resolvidos */
} carga_t;
#define CAMPOS_CARGA 4

/* Coordenador: gerencia o saco ate esvaziar e terminar todos os trabalhadores. Os resultados sao gravados em 'solucoes' na ordem original do lote. */
static void coordenador(const char *lote, char *solucoes, long n, int tam_unidade, int n_trabalhadores)
{
	long proximo = 0;  /* primeiro puzzle que ainda esta no saco */
	int  encerrados = 0;

	/* Bloco que cada trabalhador tem em maos (indice do primeiro puzzle), indexado pelo rank. */
	long *bloco_em_maos = malloc((size_t)(n_trabalhadores + 1) * sizeof(long));
	if (!bloco_em_maos) { fprintf(stderr, "erro: sem memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

	while (encerrados < n_trabalhadores) {
		MPI_Status status;

		/* Espera a proxima mensagem de QUALQUER trabalhador: atende por ordem de chegada. */
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		int origem = status.MPI_SOURCE;

		if (status.MPI_TAG == TAG_RESULTADO) {
			/* Recebe o bloco resolvido direto na sua posicao original da matriz de solucoes. */
			int bytes;
			MPI_Get_count(&status, MPI_CHAR, &bytes);
			MPI_Recv(solucoes + bloco_em_maos[origem] * LARGURA, bytes, MPI_CHAR,
			         origem, TAG_RESULTADO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(NULL, 0, MPI_CHAR, origem, TAG_PEDIDO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		/* Toda mensagem recebida e um pedido de trabalho: entrega o proximo bloco ou manda terminar. */
		if (proximo < n) {
			long qtd = (n - proximo < tam_unidade) ? n - proximo : tam_unidade;
			bloco_em_maos[origem] = proximo;
			MPI_Send(lote + proximo * LARGURA, (int)(qtd * LARGURA), MPI_CHAR,
			         origem, TAG_TRABALHO, MPI_COMM_WORLD);
			proximo += qtd;
		} else {
			MPI_Send(NULL, 0, MPI_CHAR, origem, TAG_FIM, MPI_COMM_WORLD);
			++encerrados;
		}
	}
	free(bloco_em_maos);
}

/* Trabalhador: pede trabalho, resolve o bloco, devolve, ate receber TAG_FIM. */
static carga_t trabalhador(const sdaux_t *aux, int tam_unidade)
{
	carga_t carga = {0};
	char *entrada = malloc((size_t)tam_unidade * LARGURA);
	char *saida   = malloc((size_t)tam_unidade * LARGURA);
	if (!entrada || !saida) { fprintf(stderr, "erro: sem memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

	double inicio = MPI_Wtime();
	MPI_Send(NULL, 0, MPI_CHAR, COORDENADOR, TAG_PEDIDO, MPI_COMM_WORLD);

	for (;;) {
		MPI_Status status;
		MPI_Recv(entrada, tam_unidade * LARGURA, MPI_CHAR, COORDENADOR, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == TAG_FIM) break;

		/* O ultimo bloco do saco pode ser menor que tam_unidade. */
		int bytes;
		MPI_Get_count(&status, MPI_CHAR, &bytes);
		int qtd = bytes / LARGURA;

		double t0 = MPI_Wtime();
		for (int i = 0; i < qtd; ++i)
			sd_solve(aux, entrada + i * LARGURA, saida + i * LARGURA);
		carga.tempo_calculo += MPI_Wtime() - t0;
		carga.unidades += 1;
		carga.puzzles  += qtd;

		/* Devolver o resultado e, ao mesmo tempo, pedir o proximo bloco. */
		MPI_Send(saida, bytes, MPI_CHAR, COORDENADOR, TAG_RESULTADO, MPI_COMM_WORLD);
	}
	carga.tempo_total = MPI_Wtime() - inicio;

	free(entrada); free(saida);
	return carga;
}

/* Desbalanceamento, mesma metrica do T1: (maior carga - carga media) / carga media, sobre o tempo de calculo dos trabalhadores. 0% = perfeitamente balanceado. */
static void resume_carga(const carga_t *cargas, int n_trabalhadores, double *desbalanceamento,
                         double *ocupacao_media, double *unid_min, double *unid_max)
{
	double soma = 0, maior = 0, soma_ocupacao = 0;
	*unid_min = 1e300; *unid_max = 0;
	for (int w = 0; w < n_trabalhadores; ++w) {
		const carga_t *c = &cargas[w];
		soma += c->tempo_calculo;
		if (c->tempo_calculo > maior) maior = c->tempo_calculo;
		if (c->tempo_total > 0) soma_ocupacao += c->tempo_calculo / c->tempo_total;
		if (c->unidades < *unid_min) *unid_min = c->unidades;
		if (c->unidades > *unid_max) *unid_max = c->unidades;
	}
	double media = soma / n_trabalhadores;
	*desbalanceamento = media > 0 ? (maior - media) / media : 0;
	*ocupacao_media   = soma_ocupacao / n_trabalhadores;
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int rank, np;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	const char *caminho = NULL, *saida_solucoes = NULL;
	long n_desejados = 0;
	int  repeticoes = 1, misturar = 0, tam_unidade = 16, detalhar = 0;

	/* Todos os processos recebem o mesmo argv, entao todos conhecem os mesmos parametros sem precisar de broadcast. */
	int args_ok = 1;
	for (int i = 1; i < argc; ++i) {
		if (argv[i][0] != '-' && !caminho) caminho = argv[i];
		else if (!strcmp(argv[i], "-n") && i+1 < argc) n_desejados = atol(argv[++i]);
		else if (!strcmp(argv[i], "-r") && i+1 < argc) repeticoes  = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-u") && i+1 < argc) tam_unidade = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-s") && i+1 < argc) saida_solucoes = argv[++i];
		else if (!strcmp(argv[i], "-m")) misturar = 1;
		else if (!strcmp(argv[i], "-v")) detalhar = 1;
		else args_ok = 0;
	}
	if (!caminho || !args_ok || tam_unidade < 1 || np < 2) {
		if (rank == COORDENADOR)
			fprintf(stderr,
				"uso: mpirun -np P %s <puzzles.txt> [-n N] [-r R] [-u U] [-m] [-v] [-s ARQ]   (P >= 2)\n"
				"  -n N     tamanho do lote (repete os puzzles se N > arquivo)\n"
				"  -r R     repete a medicao R vezes\n"
				"  -u U     puzzles por unidade de trabalho (padrao 16)\n"
				"  -m       embaralha o lote (seed fixo; dispersa os custos)\n"
				"  -v       imprime a carga de cada trabalhador\n"
				"  -s ARQ   grava as solucoes em ARQ\n", argv[0]);
		MPI_Finalize();
		return 1;
	}

	int n_trabalhadores = np - 1;
	sdaux_t *aux = sd_genmat();  /* cada processo gera a sua copia; e deterministica e barata */

	/* Preparacao do saco, so no coordenador e fora da regiao cronometrada. */
	long n = 0, n_unicos = 0;
	char *lote = NULL, *solucoes = NULL;
	carga_t *cargas = NULL;
	if (rank == COORDENADOR) {
		lote = carrega_lote(caminho, n_desejados, &n, &n_unicos);
		if (misturar) embaralha(lote, n, 20260909ULL);
		solucoes = calloc((size_t)n, LARGURA);
		cargas   = malloc((size_t)np * sizeof(carga_t));  /* posicao 0 (coordenador) e descartada */
		if (!solucoes || !cargas) { fprintf(stderr, "erro: sem memoria\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

		printf("# %s | puzzles=%ld (unicos=%ld%s) | ordem=%s | MPI np=%d (trabalhadores=%d) | unidade=%d\n",
		       caminho, n, n_unicos, (n > n_unicos) ? ", ciclado" : "",
		       misturar ? "embaralhada" : "por dificuldade", np, n_trabalhadores, tam_unidade);
		printf("# %-6s %12s %14s %20s %10s %10s %12s\n",
		       "rep", "tempo(s)", "puzzles/s", "checksum", "desbal(%)", "ocup(%)", "unid min-max");
		fflush(stdout);
	}

	double melhor = 0.0;
	for (int rep = 0; rep < repeticoes; ++rep) {
		carga_t minha = {0};

		/* Todos partem juntos; o tempo e medido no coordenador, do inicio ate o ultimo TAG_FIM. */
		MPI_Barrier(MPI_COMM_WORLD);
		double t0 = MPI_Wtime();

		if (rank == COORDENADOR) coordenador(lote, solucoes, n, tam_unidade, n_trabalhadores);
		else                     minha = trabalhador(aux, tam_unidade);

		double tempo = MPI_Wtime() - t0;

		/* Fora do tempo: o coordenador junta a contabilidade de carga de todos. */
		MPI_Gather(&minha, CAMPOS_CARGA, MPI_DOUBLE, cargas, CAMPOS_CARGA, MPI_DOUBLE, COORDENADOR, MPI_COMM_WORLD);

		if (rank == COORDENADOR) {
			double desbal, ocupacao, umin, umax;
			resume_carga(cargas + 1, n_trabalhadores, &desbal, &ocupacao, &umin, &umax);
			if (rep == 0 || tempo < melhor) melhor = tempo;
			printf("  %-6d %12.4f %14.0f %20llx %10.2f %10.2f %6.0f-%-6.0f\n",
			       rep + 1, tempo, n / tempo, (unsigned long long)checksum(solucoes, n),
			       100 * desbal, 100 * ocupacao, umin, umax);
			if (detalhar) {
				printf("#   %-6s %10s %10s %12s %12s %8s\n", "rank", "unidades", "puzzles", "calculo(s)", "total(s)", "ocup(%)");
				for (int w = 1; w < np; ++w)
					printf("#   %-6d %10.0f %10.0f %12.4f %12.4f %8.2f\n", w, cargas[w].unidades, cargas[w].puzzles,
					       cargas[w].tempo_calculo, cargas[w].tempo_total,
					       cargas[w].tempo_total > 0 ? 100 * cargas[w].tempo_calculo / cargas[w].tempo_total : 0);
			}
			fflush(stdout);
		}
	}

	if (rank == COORDENADOR) {
		printf("# melhor tempo: %.4f s\n", melhor);
		if (saida_solucoes) {
			FILE *fs = fopen(saida_solucoes, "w");
			if (!fs) { perror(saida_solucoes); MPI_Abort(MPI_COMM_WORLD, 1); }
			for (long i = 0; i < n; ++i) fprintf(fs, "%s\n", solucoes + i * LARGURA);
			fclose(fs);
		}
		free(lote); free(solucoes); free(cargas);
	}
	free(aux);
	MPI_Finalize();
	return 0;
}
