import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    jobs = int(header_tokens[0])
    machines = int(header_tokens[1])

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

def plotar_gantt (solucao, instancia: UPMSPInstance, titulo="Gráfico de Gantt - Escalonamento", salvar_em=None, mostrar_setup=True, mostrar_legenda_setup=True):
    """
        Plota a solução como um Gráfico de Gantt.

        Cada linha do gráfico representa uma máquina; cada barra colorida
        representa um job sendo processado (com o número do job dentro da
        barra); blocos cinza-hachurados (opcional) representam o tempo de
        setup gasto entre dois jobs consecutivos na mesma máquina.

        :param solucao: lista de listas -- solucao[m] = sequência de jobs na máquina m
        :param instancia: UPMSPInstance usada para obter os tempos de processamento/setup
        :param titulo: título do gráfico
        :param salvar_em: caminho de arquivo (ex.: "gantt.png") para salvar a figura, opcional
        :param mostrar_setup: se True, desenha o tempo de setup como bloco hachurado
        :param mostrar_legenda_setup: se True, mostra a legenda do bloco de setup
        :return: (fig, ax) para customizações extras, se necessário
        """

    M = instancia.maquinas
    fig, ax = plt.subplots(figsize=(24, 1.2 * M + 4))
    cmap = plt.get_cmap("tab20")
    cores_job = {}
    makespan_total = 0.0

    for maquina in range(M):
        jobs_na_maquina = solucao[maquina]
        tempo_atual = 0.0

        for i, job in enumerate(jobs_na_maquina):
            if job not in cores_job:
                cores_job[job] = cmap(len(cores_job) % 20)

            if i > 0 and mostrar_setup:
                job_anterior = jobs_na_maquina[i - 1]
                setup = instancia.matriz_setup[maquina][job_anterior][job]
                if setup > 0:
                    ax.barh(maquina, setup, left=tempo_atual, height=0.6,
                            color='lightgray', edgecolor='black', hatch='//', zorder=2)
                    tempo_atual += setup

            proc = instancia.matriz_processamento[job][maquina]
            ax.barh(maquina, proc, left=tempo_atual, height=0.6,
                    color=cores_job[job], edgecolor='black', zorder=2)
            ax.text(tempo_atual + proc / 2, maquina, f"J{job}",
                    ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
            tempo_atual += proc

        makespan_total = max(makespan_total, tempo_atual)

    ax.axvline(makespan_total, color='red', linestyle='--', linewidth=1.2, zorder=1)
    ax.text(makespan_total, M - 0.4, f'  Makespan = {makespan_total:.0f}',
            color='red',va='top', fontsize=9)
    ax.set_yticks(range(M))
    ax.set_yticklabels([f"Máquina {m}" for m in range(M)])
    ax.set_xlabel('Tempo')
    ax.set_title(titulo)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

    if mostrar_setup and mostrar_legenda_setup:
        setup_patch = mpatches.Patch(facecolor='lightgray', edgecolor='black', hatch='//', label='Setup')
        ax.legend(handles=[setup_patch], loc='lower right')

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=300)
        print(f"Gráfico salvo em: {salvar_em}")

    plt.show()
    return fig, ax

def plotar_convergencia_simulated_annealing (historico, titulo="Convergência do Simulated Annealing", salvar_em=None, mostrar_temperatura=True):
    """
        Plota a evolução do makespan ao longo das iterações do Simulated
        Annealing: a curva "atual" mostra a exploração (incluindo pioras
        temporariamente aceitas), e a curva "melhor encontrado" mostra o
        progresso real da otimização. Opcionalmente, sobrepõe a curva de
        resfriamento da temperatura em um eixo secundário (escala log).

        :param historico: dict retornado por
                           simulated_annealing(..., retornar_historico=True),
                           contendo as chaves 'iteracao', 'temperatura',
                           'makespan_atual' e 'melhor_makespan'
        :param titulo: título do gráfico
        :param salvar_em: caminho de arquivo (ex.: "convergencia.png") para salvar a figura, opcional
        :param mostrar_temperatura: se True, mostra a curva de temperatura em um eixo secundário
        :return: (fig, ax) para customizações extras, se necessário
        """
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.plot(historico['iteracao'], historico['makespan_atual'], color='cadetblue', linewidth=0.9, alpha=0.7, label='Makespan atual', zorder=2)
    ax.step(historico['iteracao'], historico['melhor_makespan'], color='darkblue', linewidth=1.8, where='post', label='Melhor makespan encontrado', zorder=3)

    ax.set_xlabel('Iteração')
    ax.set_ylabel('Makespan')
    ax.set_title(titulo)
    ax.grid(linestyle='--', alpha=0.4, zorder=0)

    if mostrar_temperatura:
        ax2 = ax.twinx()
        ax2.plot(historico['iteracao'], historico['temperatura'], color='firebrick', linewidth=1.5, linestyle='--', alpha=0.8, label='Temperatura', zorder=1)
        ax2.set_ylabel('Temperatura (escala log)', color='firebrick')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='firebrick')

        linhas1, rotulos1 = ax.get_legend_handles_labels()
        linhas2, rotulos2 = ax2.get_legend_handles_labels()
        ax.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc='upper right')
    else:
        ax.legend(loc='upper right')

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=300)
        print(f"Gráfico de convergência do Simulated Annealing salvo em: {salvar_em}")

    plt.show()
    return fig, ax