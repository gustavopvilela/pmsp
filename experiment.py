import time
import os
import glob
import random
import numpy as np
import csv
from utils import carregar_instancia, plotar_gantt, plotar_convergencia_simulated_annealing
from simulated_annealing import simulated_annealing

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

if __name__ == "__main__":
    executar_experimento_lote_simulated_annealing("instances", 30, "graficos")