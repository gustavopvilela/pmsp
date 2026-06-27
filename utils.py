import numpy as np
from dataclasses import dataclass

@dataclass
class UPMSPInstance:
    jobs: int
    maquinas: int
    matriz_processamento: np.ndarray # Matriz de processamento
    matriz_setup: np.ndarray  # Matriz de setup

def carregar_instancia (arquivo: str) -> UPMSPInstance:
    with open(arquivo, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    # Cabeçalho
    header_tokens = linhas[0].split()
    print(header_tokens)
    jobs = int(header_tokens[0])
    print(jobs)
    machines = int(header_tokens[1])
    print(machines)

    linha_atual = 2

    matriz_processamentos = np.zeros((jobs, machines), dtype=int)

    for j in range(jobs):
        tokens = linhas[linha_atual].split()
        if len(tokens) == machines * 2:
            tempos = [int(tokens[i]) for i in range(1, len(tokens), 2)]
        else:
            tempos = [int(t) for t in tokens]

        matriz_processamentos[j] = tempos
        linha_atual += 1

    while linha_atual < len(linhas) and linhas[linha_atual].upper() != "SSD":
        linha_atual += 1

    linha_atual += 1

    tokens_setup = []
    while linha_atual < len(linhas) and "Resources" not in linhas[linha_atual]:
        tokens_setup.extend([int(x) for x in linhas[linha_atual].split() if x.isdigit()])
        linha_atual += 1

    tamanho_esperado_setup = machines * jobs * jobs

    if len(tokens_setup) < tamanho_esperado_setup:
        raise ValueError("Erro: faltam dados de setup no arquivo lido.")

    array_setups = np.array(tokens_setup[:tamanho_esperado_setup], dtype=int)
    matriz_setups = array_setups.reshape((machines, jobs, jobs))

    return UPMSPInstance(
        jobs=jobs,
        maquinas=machines,
        matriz_processamento=matriz_processamentos,
        matriz_setup=matriz_setups
    )

if __name__ == "__main__":
    arquivo_teste = "instances/50x4_U_1_100_S_49_rep_4.txt"
    instancia = carregar_instancia(arquivo_teste)

    print(f"Jobs: {instancia.jobs} | Máquinas: {instancia.maquinas}")
    print(f"Tempo do Job 0 na Máquina 1: {instancia.matriz_processamento[0][1]}")
    print(f"Setup na Máquina 0 (Transição do Job 2 -> Job 5): {instancia.matriz_setup[0][2][5]}")