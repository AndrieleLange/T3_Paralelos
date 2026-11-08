#!/bin/bash
# Medicoes de desempenho. Ajuste conforme o gerenciador de jobs da Atlantica
# (por exemplo, coloque este script dentro de um job com 2 nos exclusivos).
# Uso: ./run.sh [N] [repeticoes]
# Modos: 0 = D&C simples, 1 = versao do artigo (balanceada)
N=${1:-100000}
REP=${2:-3}
PROCS_BASE="1 3 7 15 31"
PROCS_EXTRA="63 127 255 511 1023"
PROCS_POT2="2 4 8 16 32 64 128"     # potencias de 2 (o modo 1 so balanceia assim)

for modo in 0 1; do
  for p in $PROCS_BASE $PROCS_POT2 $PROCS_EXTRA; do
    for i in $(seq 1 $REP); do
      # --oversubscribe: necessario quando p > numero de slots (HT / muitos processos)
      # --map-by node: espalha os processos entre os nos alocados
      mpirun --oversubscribe --map-by node -np $p ./bs_mpi $N $modo
    done
  done
done | tee resultados_N${N}.txt
