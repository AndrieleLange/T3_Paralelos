
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 40
#define DEBUG 1

void bs(int n, int *vetor)
{
    int c = 0;
    int d;
    int troca;
    int trocou = 1;

    while (c < (n - 1) && trocou)
    {
        trocou = 0;

        for (d = 0; d < n - c - 1; d++)
        {
            if (vetor[d] > vetor[d + 1])
            {
                troca = vetor[d];
                vetor[d] = vetor[d + 1];
                vetor[d + 1] = troca;

                trocou = 1;
            }
        }

        c++;
    }
}

int main()
{
    int vetor[ARRAY_SIZE];
    int i;

    // Inicializa o vetor em ordem reversa
    for (i = 0; i < ARRAY_SIZE; i++)
    {
        vetor[i] = ARRAY_SIZE - i;
    }

    #ifdef DEBUG
    printf("\nVetor original:\n");

    for (i = 0; i < ARRAY_SIZE; i++)
    {
        printf("[%03d] ", vetor[i]);
    }

    printf("\n");
    #endif

    // Ordenação sequencial
    bs(ARRAY_SIZE, vetor);

    #ifdef DEBUG
    printf("\nVetor ordenado:\n");

    for (i = 0; i < ARRAY_SIZE; i++)
    {
        printf("[%03d] ", vetor[i]);
    }

    printf("\n");
    #endif

    return 0;
}
