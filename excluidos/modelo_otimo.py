import time
import glob
import os
import numpy as np
from ortools.linear_solver import pywraplp
from utils import carregar_instancia, plotar_gantt


def resolver_pmsp_otimo(caminho_instancia, tempo_limite_minutos=10):
    inst = carregar_instancia(caminho_instancia)
    num_jobs = inst.jobs
    num_maquinas = inst.maquinas
    p = inst.matriz_processamento
    s = inst.matriz_setup

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Solver SCIP nao encontrado.")
        return None, None, None, None

    solver.SetTimeLimit(tempo_limite_minutos * 60 * 1000)
    M_grande = int(np.sum(p) + np.sum(s))

    x = {}
    for k in range(num_maquinas):
        x[k] = {}
        for i in range(num_jobs + 1):
            x[k][i] = {}
            for j in range(num_jobs):
                if i != j:
                    x[k][i][j] = solver.IntVar(0, 1, f'x_m{k}_i{i}_j{j}')

    c = [solver.NumVar(0.0, M_grande, f'c_{j}') for j in range(num_jobs)]
    cmax = solver.NumVar(0.0, M_grande, 'cmax')

    for j in range(num_jobs):
        solver.Add(
            sum(x[k][i][j] for k in range(num_maquinas) for i in range(num_jobs + 1) if i != j) == 1
        )

    for k in range(num_maquinas):
        solver.Add(sum(x[k][num_jobs][j] for j in range(num_jobs)) <= 1)

        for j in range(num_jobs):
            entradas = sum(x[k][i][j] for i in range(num_jobs + 1) if i != j)
            saidas = sum(x[k][j][h] for h in range(num_jobs) if h != j)
            solver.Add(saidas <= entradas)

    for k in range(num_maquinas):
        for j in range(num_jobs):
            solver.Add(c[j] >= int(p[j][k]) - M_grande * (1 - x[k][num_jobs][j]))
            for i in range(num_jobs):
                if i != j:
                    setup_time = int(s[k][i][j])
                    proc_time = int(p[j][k])
                    solver.Add(c[j] >= c[i] + setup_time + proc_time - M_grande * (1 - x[k][i][j]))

    for j in range(num_jobs):
        solver.Add(cmax >= c[j])

    solver.Minimize(cmax)

    status = solver.Solve()

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        # --- MUDANÇA AQUI: Capturando o valor e o status ---
        makespan_valor = cmax.solution_value()
        is_optimal = (status == pywraplp.Solver.OPTIMAL)

        solucao_final = [[] for _ in range(num_maquinas)]
        for k in range(num_maquinas):
            job_atual = -1
            for j in range(num_jobs):
                if x[k][num_jobs][j].solution_value() > 0.5:
                    job_atual = j
                    break

            while job_atual != -1:
                solucao_final[k].append(job_atual)
                proximo_job = -1
                for j in range(num_jobs):
                    if job_atual != j and x[k][job_atual][j].solution_value() > 0.5:
                        proximo_job = j
                        break
                job_atual = proximo_job

        # --- MUDANÇA AQUI: Retornando também o makespan e o status ---
        return solucao_final, inst, makespan_valor, is_optimal
    else:
        return None, None, None, None


if __name__ == "__main__":
    arquivos_instancias = glob.glob("instances/*.txt")
    arquivos_instancias.sort()

    arquivo_saida = "resultados_otimos.txt"
    tempo_limite = 10  # minutos

    with open(arquivo_saida, "w") as f:
        f.write("RELATORIO DE RESULTADOS - MODELO OTIMO (SCIP)\n")
        f.write("-" * 75 + "\n")
        # Criando um cabeçalho organizado em colunas
        f.write(f"{'Instancia':<30} | {'Status':<15} | {'Makespan':<10} | {'Tempo (s)':<10}\n")
        f.write("-" * 75 + "\n")

    for caminho in arquivos_instancias:
        nome_arquivo = os.path.basename(caminho)
        print(f"\nResolvendo: {nome_arquivo}...")

        inicio = time.time()
        # --- MUDANÇA AQUI: Recebendo os 4 valores retornados ---
        solucao, instancia, makespan, is_optimal = resolver_pmsp_otimo(caminho, tempo_limite_minutos=tempo_limite)
        fim = time.time()

        tempo_gasto = fim - inicio

        with open(arquivo_saida, "a") as f:
            if solucao:
                # Formata o status para ficar fácil de ler no TXT
                status_str = "OTIMO" if is_optimal else "VIAVEL"
                linha = f"{nome_arquivo:<30} | {status_str:<15} | {makespan:<10.2f} | {tempo_gasto:<10.2f}\n"
                f.write(linha)
                print(f"-> Concluido! Makespan: {makespan} ({status_str})")
                plotar_gantt(solucao, instancia, titulo=f"Otimo - {nome_arquivo}")
            else:
                linha = f"{nome_arquivo:<30} | {'SEM SOLUCAO':<15} | {'-':<10} | {tempo_gasto:<10.2f}\n"
                f.write(linha)
                print("-> Sem solucao encontrada no tempo limite.")

