import csv
import os
import numpy as np

# Certifique-se de que os nomes importados batem com os nomes das funções no seu arquivo 'graficos_pmsp.py'
from graficos_pmsp import (
    gerar_grafico_limites_vs_resultados,
    gerar_grafico_barras_runtime,
    gerar_grafico_intervalo_confianca,
    gerar_ic_por_tamanho
)
from teste_hipotese_pmsp import comparar_duas_metaheuristicas


def ler_resultados_exatos(caminho_txt):
    """Lê o TXT do CP-SAT e extrai o Limite Superior (UB) e o Inferior (LB)."""
    dados_exatos = {}
    if not os.path.exists(caminho_txt):
        print(f"Aviso: Arquivo {caminho_txt} não encontrado.")
        return dados_exatos

    with open(caminho_txt, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
        for linha in linhas:
            if ".txt" in linha and "|" in linha:
                partes = [p.strip() for p in linha.split('|')]
                instancia = partes[0]
                try:
                    ub = float(partes[2])  # Extrai o UB
                    lb = float(partes[3])  # Extrai o LB
                    dados_exatos[instancia] = {'UB': ub, 'LB': lb}
                except ValueError:
                    pass
    return dados_exatos


def ler_resultados_metaheuristica(caminho_csv):
    """Lê o CSV gerado pelo experiment.py extraindo Best, Average, Worst e Tempo_Medio."""
    dados_meta = {}
    if not os.path.exists(caminho_csv):
        print(f"Aviso: Arquivo {caminho_csv} não encontrado.")
        return dados_meta

    # utf-8-sig previne problemas se o CSV foi salvo com formatação do Excel
    with open(caminho_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instancia = row['Instancia']
            dados_meta[instancia] = {
                'Best': float(row['Best']),
                'Average': float(row['Average']),
                'Worst': float(row['Worst']),
                'Tempo_Medio': float(row['Tempo_Medio'])
            }

    return dados_meta


if __name__ == "__main__":
    print("=" * 60)
    print("INICIANDO ANÁLISE DE DADOS E TESTES ESTATÍSTICOS")
    print("=" * 60)

    # 1. LER OS ARQUIVOS
    exatos = ler_resultados_exatos("resultados_cp_sat.txt")
    sa_dados = ler_resultados_metaheuristica("sa_resumo_estatistico.csv")
    grasp_dados = ler_resultados_metaheuristica("grasp_resumo_estatistico.csv")

    # 2. CRUZAR OS DADOS E CALCULAR GAPS
    instancias_comuns = []
    gaps_sa, gaps_grasp = [], []
    tempos_sa, tempos_grasp = [], []

    # Listas adicionais para o gráfico de Limites vs Resultados
    lb_lista, ub_lista = [], []
    best_sa_lista, avg_sa_lista, worst_sa_lista = [], [], []
    best_grasp_lista, avg_grasp_lista, worst_grasp_lista = [], [], []

    for inst in sa_dados.keys():
        # Verifica se a instância existe nos 3 arquivos e se o CP-SAT achou o gabarito (LB > 0)
        if inst in grasp_dados and inst in exatos and exatos[inst]['LB'] > 0:
            lb_exato = exatos[inst]['LB']
            ub_exato = exatos[inst]['UB']

            # SA
            best_sa = sa_dados[inst]['Best']
            avg_sa = sa_dados[inst]['Average']
            worst_sa = sa_dados[inst]['Worst']
            tempo_sa = sa_dados[inst]['Tempo_Medio']

            # GRASP
            best_grasp = grasp_dados[inst]['Best']
            avg_grasp = grasp_dados[inst]['Average']
            worst_grasp = grasp_dados[inst]['Worst']
            tempo_grasp = grasp_dados[inst]['Tempo_Medio']

            # Cálculo do Gap % em relação ao UB (ou LB, dependendo da sua preferência)
            # Vamos calcular em relação ao UB para o teste de Wilcoxon / Intervalo de Confiança
            gap_sa = ((best_sa - ub_exato) / ub_exato) * 100
            gap_grasp = ((best_grasp - ub_exato) / ub_exato) * 100

            # Preenchimento das listas
            instancias_comuns.append(inst)
            lb_lista.append(lb_exato)
            ub_lista.append(ub_exato)

            best_sa_lista.append(best_sa)
            avg_sa_lista.append(avg_sa)
            worst_sa_lista.append(worst_sa)
            tempos_sa.append(tempo_sa)
            gaps_sa.append(gap_sa)

            best_grasp_lista.append(best_grasp)
            avg_grasp_lista.append(avg_grasp)
            worst_grasp_lista.append(worst_grasp)
            tempos_grasp.append(tempo_grasp)
            gaps_grasp.append(gap_grasp)

    if len(instancias_comuns) == 0:
        print("Erro: Nenhuma instância em comum encontrada entre os 3 arquivos.")
        print("Verifique se os nomes das instâncias batem exatamente no TXT e nos CSVs.")
        exit()

    print(f"Foram cruzados dados de {len(instancias_comuns)} instâncias.\n")

    # Limitador para gráficos de barras/pontos para não poluir visualmente (opcional)
    limite = min(15, len(instancias_comuns))

    # 3. GERAR GRÁFICOS
    print("Gerando Gráfico de Limites vs Resultados (SA)...")
    gerar_grafico_limites_vs_resultados(
        instancias_comuns[:limite], lb_lista[:limite], ub_lista[:limite],
        best_sa_lista[:limite], avg_sa_lista[:limite], worst_sa_lista[:limite],
        "Simulated Annealing"
    )

    print("Gerando Gráfico de Limites vs Resultados (GRASP)...")
    gerar_grafico_limites_vs_resultados(
        instancias_comuns[:limite], lb_lista[:limite], ub_lista[:limite],
        best_grasp_lista[:limite], avg_grasp_lista[:limite], worst_grasp_lista[:limite],
        "GRASP-PR"
    )

    print("Gerando Gráfico de Runtimes...")
    gerar_grafico_barras_runtime(
        tempos_sa[:limite], tempos_grasp[:limite],
        instancias_comuns[:limite], "Simulated Annealing", "GRASP-PR"
    )

    #print("Gerando Intervalo de Confiança...")
    #gerar_grafico_intervalo_confianca(gaps_sa, gaps_grasp, "Simulated Annealing", "GRASP-PR")
    print("Gerando Intervalo de Confiança Agrupado...")
    gerar_ic_por_tamanho(instancias_comuns, gaps_sa, gaps_grasp, "Simulated Annealing", "GRASP-PR")

    # 4. TESTE DE HIPÓTESE
    print("Rodando Teste de Wilcoxon...")
    comparar_duas_metaheuristicas(gaps_sa, gaps_grasp, "Simulated Annealing", "GRASP-PR")

    print("\nAnálise concluída com sucesso! Verifique as imagens salvas na pasta.")