import csv
import os
import numpy as np
from graficos_pmsp import gerar_boxplot_gaps, gerar_grafico_barras_runtime
from teste_hipotese_pmsp import comparar_duas_metaheuristicas


def ler_resultados_exatos(caminho_txt):
    """Lê o TXT do CP-SAT e extrai o Limite Matemático (LB) de cada instância."""
    dados_exatos = {}
    if not os.path.exists(caminho_txt):
        print(f"Aviso: Arquivo {caminho_txt} não encontrado.")
        return dados_exatos

    with open(caminho_txt, 'r') as f:
        linhas = f.readlines()
        for linha in linhas:
            # Pula cabeçalhos e linhas de separação
            if ".txt" in linha and "|" in linha:
                partes = [p.strip() for p in linha.split('|')]
                instancia = partes[0]
                status = partes[1]
                # Pega o LB (coluna 3 do seu print: 0=Instancia, 1=Status, 2=UB, 3=LB)
                try:
                    lb = float(partes[3])
                    dados_exatos[instancia] = lb
                except ValueError:
                    # Se não achou LB (ex: estourou tempo sem achar nada), ignora
                    pass
    return dados_exatos


def ler_resultados_metaheuristica(caminho_csv):
    """Lê o CSV gerado pelo experiment.py e extrai Makespan Médio e Tempo Médio."""
    dados_meta = {}
    if not os.path.exists(caminho_csv):
        print(f"Aviso: Arquivo {caminho_csv} não encontrado.")
        return dados_meta

    with open(caminho_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instancia = row['Instancia']
            # Vamos usar o 'Average' (Média das 30 rodadas) para o teste justo
            makespan_medio = float(row['Average'])
            tempo_medio = float(row['Tempo_Medio'])
            dados_meta[instancia] = {'makespan': makespan_medio, 'tempo': tempo_medio}

    return dados_meta


if __name__ == "__main__":
    print("=" * 60)
    print("INICIANDO ANÁLISE DE DADOS E TESTES ESTATÍSTICOS")
    print("=" * 60)

    # 1. LER OS ARQUIVOS
    # Ajuste os nomes dos arquivos conforme os que você gerou!
    exatos = ler_resultados_exatos("resultados_cp_sat.txt")
    sa_dados = ler_resultados_metaheuristica("sa_resumo_estatistico.csv")

    # OBS: Você precisa adaptar seu experiment.py para rodar o GRASP e gerar esse CSV
    grasp_dados = ler_resultados_metaheuristica("grasp_resumo_estatistico.csv")

    # 2. CRUZAR OS DADOS E CALCULAR GAPS
    instancias_comuns = []
    gaps_sa = []
    gaps_grasp = []
    tempos_sa = []
    tempos_grasp = []

    for inst in sa_dados.keys():
        # Só analisa se a instância rodou no SA, no GRASP e se o modelo Exato achou o Gabarito (LB > 0)
        if inst in grasp_dados and inst in exatos and exatos[inst] > 0:
            lb_exato = exatos[inst]

            # Cálculo do Gap %: ((Makespan_Meta - LB_Exato) / LB_Exato) * 100
            gap_sa = ((sa_dados[inst]['makespan'] - lb_exato) / lb_exato) * 100
            gap_grasp = ((grasp_dados[inst]['makespan'] - lb_exato) / lb_exato) * 100

            instancias_comuns.append(inst)
            gaps_sa.append(gap_sa)
            gaps_grasp.append(gap_grasp)
            tempos_sa.append(sa_dados[inst]['tempo'])
            tempos_grasp.append(grasp_dados[inst]['tempo'])

    if len(instancias_comuns) == 0:
        print("Erro: Nenhuma instância em comum encontrada entre os 3 arquivos.")
        exit()

    print(f"Foram cruzados dados de {len(instancias_comuns)} instâncias.\n")

    # 3. GERAR GRÁFICOS
    print("Gerando Boxplot de Gaps...")
    gerar_boxplot_gaps(gaps_sa, gaps_grasp, "Simulated Annealing", "GRASP-PR")

    print("Gerando Gráfico de Runtimes...")
    # Se tiver muitas instâncias, pega só as 15 primeiras para o gráfico de barras não ficar ilegível
    limite = min(15, len(instancias_comuns))
    gerar_grafico_barras_runtime(tempos_sa[:limite], tempos_grasp[:limite],
                                 instancias_comuns[:limite], "Simulated Annealing", "GRASP-PR")

    # 4. TESTE DE HIPÓTESE
    print("Rodando Teste de Wilcoxon...")
    comparar_duas_metaheuristicas(gaps_sa, gaps_grasp, "Simulated Annealing", "GRASP-PR")

    print("\nAnálise concluída com sucesso! Verifique as imagens salvas na pasta.")