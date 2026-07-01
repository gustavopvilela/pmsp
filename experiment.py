import time
import os
import glob
import random
import numpy as np
import csv
import copy
from utils import carregar_instancia, plotar_gantt, plotar_convergencia_simulated_annealing
from simulated_annealing import simulated_annealing
from grasp_pathrelinking import grasp_path_relinking

def executar_experimento_lote_simulated_annealing (pasta_instancias="instances", num_rodadas=30, pasta_graficos="graficos"):
    padrao_busca = os.path.join(pasta_instancias, "*.txt")
    arquivos = glob.glob(padrao_busca)

    if not arquivos:
        print(f"Erro: nenhum arquivo .txt encontrado na pasta '{pasta_instancias}'.")
        return

    os.makedirs(pasta_graficos, exist_ok=True)

    print(f"Experimentos em lote iniciados: {len(arquivos)} instâncias encontradas.")
    print("=" * 60)

    with open('sa_dados_brutos.csv', 'w', newline='') as f_bruto, \
         open('sa_resumo_estatistico.csv', 'w', newline='') as f_resumo:
        writer_bruto = csv.writer(f_bruto)
        writer_resumo = csv.writer(f_resumo)
        writer_bruto.writerow(["Instancia", "Rodada", "Semente", "Makespan", "Tempo_Segundos"])
        writer_resumo.writerow(["Instancia", "Best", "Average", "Worst", "StdDev", "Tempo_Medio"])

        for arquivo in arquivos:
            nome_instancia = os.path.basename(arquivo)
            nome_base, _ = os.path.splitext(nome_instancia)
            print(f"Processando: {nome_instancia} ({num_rodadas} rodadas)")
            instancia = carregar_instancia(arquivo)
            resultados_makespan = []
            resultados_tempo = []
            melhor_makespan_instancia = np.inf
            melhor_solucao_instancia = None
            melhor_historico_instacia = None

            for i in range(num_rodadas):
                semente_atual = i + 1
                random.seed(semente_atual)
                np.random.seed(semente_atual)
                inicio_tempo = time.perf_counter()
                solucao, makespan, historico = simulated_annealing(
                    instancia=instancia,
                    temp_inicial=5000,
                    iteracoes_por_temp=500,
                    tempo_limite_segundos=300
                )
                fim_tempo = time.perf_counter()
                tempo_execucao = fim_tempo - inicio_tempo
                resultados_makespan.append(makespan)
                resultados_tempo.append(tempo_execucao)
                writer_bruto.writerow([nome_instancia, i + 1, semente_atual, makespan, tempo_execucao])

                if makespan < melhor_makespan_instancia:
                    melhor_makespan_instancia = makespan
                    melhor_solucao_instancia = solucao
                    melhor_historico_instacia = historico

            best = np.min(resultados_makespan)
            worst = np.max(resultados_makespan)
            avg = np.mean(resultados_makespan)
            std = np.std(resultados_makespan)
            tempo_med = np.mean(resultados_tempo)
            writer_resumo.writerow([nome_instancia, best, avg, worst, std, tempo_med])
            print(f" -> Concluído! Média: {avg:.1f} | Desvio: {std:.2f} | Tempo Médio: {tempo_med:.3f}s")
            print("-" * 60)

            caminho_gantt = os.path.join(pasta_graficos, f"{nome_base}_gantt.png")
            caminho_convergencia = os.path.join(pasta_graficos, f"{nome_base}_convergencia.png")
            plotar_gantt(
                melhor_solucao_instancia,
                instancia,
                titulo=f"{nome_instancia} (makespan = {melhor_makespan_instancia})",
                salvar_em=caminho_gantt,
                mostrar=False
            )
            plotar_convergencia_simulated_annealing(
                melhor_historico_instacia,
                titulo=f"Convergência - {nome_instancia}",
                salvar_em=caminho_convergencia,
                mostrar=False
            )
            print(f" -> Gráficos salvos: {caminho_gantt} | {caminho_convergencia}")
            print("-" * 60)

    print("\nTESTE EM LOTE FINALIZADO!")
    print("Os arquivos 'sa_dados_brutos.csv' e 'sa_resumo_estatistico.csv' foram gerados com sucesso.")

def executar_experimento_simulated_annealing (caminho_instancia, num_rodadas=30, arquivo_saida="resultados_sa.csv"):
    """
    Executa o Simulated Annealing múltiplas vezes com controle de semente,
    calcula as estatísticas e exporta para CSV.
    """
    print(f"Carregando instância: {caminho_instancia}")
    instancia = carregar_instancia(caminho_instancia)
    resultados_makespan = []
    resultados_tempo = []

    print(f"Iniciando {num_rodadas} independentes.")
    print("-" * 50)
    print(f"{'Rodada':<10} | {'Semente':<10} | {'Makespan':<12} | {'Tempo (s)':<10}")
    print("-" * 50)

    for i in range(num_rodadas):
        semente_atual = i + 1
        random.seed(semente_atual)
        np.random.seed(semente_atual)
        tempo_inicio = time.perf_counter()
        solucao, makespan, historico = simulated_annealing(
            instancia=instancia,
            temp_inicial=5000,
            iteracoes_por_temp=500,
            tempo_limite_segundos=300
        )
        tempo_fim = time.perf_counter()
        tempo_execucao = tempo_fim - tempo_inicio
        resultados_makespan.append(makespan)
        resultados_tempo.append(tempo_execucao)
        print(f"{i + 1:<10} | {semente_atual:<10} | {makespan:<12.1f} | {tempo_execucao:<10.3f}")

    print("-" * 50)

    melhor_valor = np.min(resultados_makespan)
    pior_valor = np.max(resultados_makespan)
    media_valor = np.mean(resultados_makespan)
    desvio_padrao = np.std(resultados_makespan)
    tempo_medio = np.mean(resultados_tempo)

    print("\nRESUMO ESTATÍSTICO:")
    print(f"Best (Melhor):   {melhor_valor:.1f}")
    print(f"Average (Média): {media_valor:.1f}")
    print(f"Worst (Pior):    {pior_valor:.1f}")
    print(f"Std Dev (Desvio):{desvio_padrao:.2f}")
    print(f"Tempo Médio:     {tempo_medio:.3f} s")

    with open(arquivo_saida, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rodada", "Semente", "Makespan", "Tempo (s)"])

        for i in range(num_rodadas):
            writer.writerow([i + 1, i + 1, resultados_makespan[i], resultados_tempo[i]])

        writer.writerow([])
        writer.writerow(["Estatística", "Valor"])
        writer.writerow(["Best", melhor_valor])
        writer.writerow(["Average", media_valor])
        writer.writerow(["Worst", pior_valor])
        writer.writerow(["StdDev", desvio_padrao])
        writer.writerow(["Tempo_Medio", tempo_medio])

    print(f"\nDados exportados com sucesso para: {arquivo_saida}")

def executar_experimento_lote_grasp_pr(pasta_instancias="instances", num_rodadas=30, pasta_graficos="graficos"):
    padrao_busca = os.path.join(pasta_instancias, "*.txt")
    arquivos = glob.glob(padrao_busca)

    if not arquivos:
        print(f"Erro: nenhum arquivo .txt encontrado na pasta '{pasta_instancias}'.")
        return

    os.makedirs(pasta_graficos, exist_ok=True)

    print(f"Experimentos em lote GRASP+PR iniciados: {len(arquivos)} instâncias encontradas.")
    print("=" * 60)

    # Nomes dos arquivos de saída específicos para o GRASP
    with open('grasp_dados_brutos.csv', 'w', newline='') as f_bruto, \
            open('grasp_resumo_estatistico.csv', 'w', newline='') as f_resumo:
        writer_bruto = csv.writer(f_bruto)
        writer_resumo = csv.writer(f_resumo)
        writer_bruto.writerow(["Instancia", "Rodada", "Semente", "Makespan", "Tempo_Segundos"])
        writer_resumo.writerow(["Instancia", "Best", "Average", "Worst", "StdDev", "Tempo_Medio"])

        for arquivo in arquivos:
            nome_instancia = os.path.basename(arquivo)
            nome_base, _ = os.path.splitext(nome_instancia)
            print(f"Processando: {nome_instancia} ({num_rodadas} rodadas)")

            instancia = carregar_instancia(arquivo)
            resultados_makespan = []
            resultados_tempo = []
            melhor_makespan_instancia = np.inf
            melhor_solucao_instancia = None

            for i in range(num_rodadas):
                semente_atual = i + 1
                random.seed(semente_atual)
                np.random.seed(semente_atual)

                inicio_tempo = time.perf_counter()

                # Chamada do GRASP+PR corrigida
                solucao, makespan = grasp_path_relinking(
                    instancia=instancia,
                    max_iter=20,
                    alpha=0.3,
                    max_elite=10
                )

                fim_tempo = time.perf_counter()
                tempo_execucao = fim_tempo - inicio_tempo

                resultados_makespan.append(makespan)
                resultados_tempo.append(tempo_execucao)
                writer_bruto.writerow([nome_instancia, i + 1, semente_atual, makespan, tempo_execucao])

                # Atualiza a melhor solução global para esta instância
                if makespan < melhor_makespan_instancia:
                    melhor_makespan_instancia = makespan
                    melhor_solucao_instancia = copy.deepcopy(solucao)

            # Cálculos estatísticos da instância
            best = np.min(resultados_makespan)
            worst = np.max(resultados_makespan)
            avg = np.mean(resultados_makespan)
            std = np.std(resultados_makespan)
            tempo_med = np.mean(resultados_tempo)
            writer_resumo.writerow([nome_instancia, best, avg, worst, std, tempo_med])

            print(f" -> Concluído! Média: {avg:.1f} | Desvio: {std:.2f} | Tempo Médio: {tempo_med:.3f}s")
            print("-" * 60)

            # Plota apenas o Gantt da melhor rodada
            caminho_gantt = os.path.join(pasta_graficos, f"{nome_base}_GRASP_gantt.png")
            plotar_gantt(
                melhor_solucao_instancia,
                instancia,
                titulo=f"GRASP+PR: {nome_instancia} (makespan = {melhor_makespan_instancia})",
                salvar_em=caminho_gantt,
                mostrar=False
            )
            print(f" -> Gráfico salvo: {caminho_gantt}")
            print("-" * 60)

    print("\nTESTE EM LOTE GRASP+PR FINALIZADO!")
    print("Os arquivos 'grasp_dados_brutos.csv' e 'grasp_resumo_estatistico.csv' foram gerados com sucesso.")


def executar_experimento_grasp_path_relinking(caminho_instancia, num_rodadas=30, arquivo_saida="resultados_grasp.csv"):
    """
    Executa o GRASP+PR múltiplas vezes com controle de semente,
    calcula as estatísticas e exporta para CSV.
    """
    print(f"Carregando instância: {caminho_instancia}")
    instancia = carregar_instancia(caminho_instancia)
    resultados_makespan = []
    resultados_tempo = []

    print(f"Iniciando {num_rodadas} rodadas independentes.")
    print("-" * 50)
    print(f"{'Rodada':<10} | {'Semente':<10} | {'Makespan':<12} | {'Tempo (s)':<10}")
    print("-" * 50)

    for i in range(num_rodadas):
        semente_atual = i + 1
        random.seed(semente_atual)
        np.random.seed(semente_atual)

        tempo_inicio = time.perf_counter()

        # Chamada corrigida para o algoritmo correto
        solucao, makespan = grasp_path_relinking(
            instancia=instancia,
            max_iter=20
            ,
            alpha=0.3,
            max_elite=10
        )

        tempo_fim = time.perf_counter()
        tempo_execucao = tempo_fim - tempo_inicio

        resultados_makespan.append(makespan)
        resultados_tempo.append(tempo_execucao)
        print(f"{i + 1:<10} | {semente_atual:<10} | {makespan:<12.1f} | {tempo_execucao:<10.3f}")

    print("-" * 50)

    melhor_valor = np.min(resultados_makespan)
    pior_valor = np.max(resultados_makespan)
    media_valor = np.mean(resultados_makespan)
    desvio_padrao = np.std(resultados_makespan)
    tempo_medio = np.mean(resultados_tempo)

    print("\nRESUMO ESTATÍSTICO GRASP+PR:")
    print(f"Best (Melhor):   {melhor_valor:.1f}")
    print(f"Average (Média): {media_valor:.1f}")
    print(f"Worst (Pior):    {pior_valor:.1f}")
    print(f"Std Dev (Desvio):{desvio_padrao:.2f}")
    print(f"Tempo Médio:     {tempo_medio:.3f} s")

    with open(arquivo_saida, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rodada", "Semente", "Makespan", "Tempo (s)"])

        for i in range(num_rodadas):
            writer.writerow([i + 1, i + 1, resultados_makespan[i], resultados_tempo[i]])

        writer.writerow([])
        writer.writerow(["Estatística", "Valor"])
        writer.writerow(["Best", melhor_valor])
        writer.writerow(["Average", media_valor])
        writer.writerow(["Worst", pior_valor])
        writer.writerow(["StdDev", desvio_padrao])
        writer.writerow(["Tempo_Medio", tempo_medio])

    print(f"\nDados exportados com sucesso para: {arquivo_saida}")


if __name__ == "__main__":
    # Aqui você pode escolher qual experimento rodar comentando/descomentando as linhas:

    # 1. Executar Lote Simulated Annealing
    # executar_experimento_lote_simulated_annealing("instances", 30, "graficos")

    # 2. Executar Lote GRASP+PR
    executar_experimento_lote_grasp_pr("instances", 30, "graficos")
