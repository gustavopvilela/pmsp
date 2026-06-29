import os
import glob
import time
from ortools.sat.python import cp_model
from utils import carregar_instancia

def resolver_modelo_exato(instancia, tempo_limite_segundos):
    """
    Usa o motor CP-SAT do OR-Tools para provar o Ótimo Global ou
    encontrar o Lower Bound (LB) da instância.
    """
    model = cp_model.CpModel()

    N = instancia.jobs
    M = instancia.maquinas

    #toda máquina "sai" deste nó, processa os jobs, e "volta" para ele.
    NO_FICTICIO = N

    #active[m, j]: True se o job j for alocado na máquina m
    active = {}
    for m in range(M):
        for j in range(N):
            active[m, j] = model.NewBoolVar(f'active_m{m}_j{j}')

    #restrição: todo job j DEVE ser feito em exatamente uma máquina
    for j in range(N):
        model.AddExactlyOne([active[m, j] for m in range(M)])

    #criação dos Arcos (transições de sequência)
    arcs = {}
    arc_costs = {}

    for m in range(M):
        arcs[m] = []
        arc_costs[m] = []

        for i in range(N + 1):
            for j in range(N + 1):
                lit = model.NewBoolVar(f'arc_m{m}_i{i}_j{j}')

                if i == j:
                    if i == NO_FICTICIO:
                        #se a máquina ficar totalmente vazia, o nó fictício aponta pra ele mesmo
                        arcs[m].append((i, j, lit))
                        arc_costs[m].append(0)
                    else:
                        #regra de circuito: se o job 'i' NÃO está na máquina 'm', ele deve fazer um
                        #auto-loop (transição fantasma) de custo zero.
                        model.Add(lit == active[m, i].Not())
                        arcs[m].append((i, j, lit))
                        arc_costs[m].append(0)

                elif i == NO_FICTICIO:
                    #início da fila (primeiro job da máquina) -> paga só o processamento
                    arcs[m].append((i, j, lit))
                    arc_costs[m].append(int(instancia.matriz_processamento[j][m]))
                    model.AddImplication(lit, active[m, j])  #se usou o arco, o job está na máquina

                elif j == NO_FICTICIO:
                    #fim da fila (último job -> volta pro ocioso) -> custo zero
                    arcs[m].append((i, j, lit))
                    arc_costs[m].append(0)
                    model.AddImplication(lit, active[m, i])

                else:
                    #miolo da fila (Job i -> Job j) -> paga setup + processamento do próximo
                    custo = int(instancia.matriz_setup[m][i][j] + instancia.matriz_processamento[j][m])
                    arcs[m].append((i, j, lit))
                    arc_costs[m].append(custo)
                    model.AddImplication(lit, active[m, i])
                    model.AddImplication(lit, active[m, j])

        #obriga que as transições formem uma rota fechada e sem sub-rotas
        model.AddCircuit(arcs[m])

    #cálculo do tempo e Função Objetivo (Makespan)
    makespan = model.NewIntVar(0, 9999999, 'makespan')
    tempos_maquinas = []

    for m in range(M):
        tempo_m = model.NewIntVar(0, 9999999, f'tempo_m{m}')
        #o tempo da máquina é a soma dos custos de todos os arcos ativos nela
        termos_custo = [lit * custo for (i, j, lit), custo in zip(arcs[m], arc_costs[m]) if custo > 0]
        model.Add(tempo_m == sum(termos_custo))
        tempos_maquinas.append(tempo_m)

    #o Makespan é o gargalo (o maior tempo entre as máquinas)
    model.AddMaxEquality(makespan, tempos_maquinas)

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = tempo_limite_segundos
    solver.parameters.log_search_progress = True  #mostra a árvore de busca no console

    print(f"\nIniciando o Google OR-Tools CP-SAT (Limite: {tempo_limite_segundos}s)...")
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n" + "=" * 45)
        print("RESULTADO DO MODELO EXATO")
        print("=" * 45)
        print(f"Status Final: {solver.StatusName(status)}")
        ub = solver.ObjectiveValue()
        print(f"Solução Encontrada (Upper Bound): {ub}")

        lb = solver.BestObjectiveBound()
        print(f"Limite Matemático (Lower Bound) : {lb}")

        gap = ((solver.ObjectiveValue() - lb) / lb) * 100
        print(f"Margem de erro do OR-Tools      : {gap:.2f}%\n")
        return solver.StatusName(status), ub, lb, gap
    else:
        print("O solver não conseguiu encontrar uma solução viável no tempo limite.")
        return "SEM_SOLUCAO", None, None, None

# --- Como Rodar ---
if __name__ == "__main__":
    #caminho onde as instâncias estão guardadas
    arquivos_instancias = glob.glob("instances/*.txt")
    arquivos_instancias.sort()

    arquivo_saida = "resultados_cp_sat.txt"
    tempo_limite = 300  # 5 minutos = 300 segundos

    #prepara o arquivo TXT criando o cabeçalho
    with open(arquivo_saida, "w") as f:
        f.write("RELATORIO DE RESULTADOS - MODELO EXATO (CP-SAT)\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Instancia':<30} | {'Status':<15} | {'UB (Makespan)':<15} | {'LB (Gabarito)':<15} | {'Gap (%)':<10}\n")
        f.write("-" * 85 + "\n")

    #percorre cada arquivo de instância
    for caminho in arquivos_instancias:
        nome_arquivo = os.path.basename(caminho)
        print(f"\n[{time.strftime('%H:%M:%S')}] Começando a resolver: {nome_arquivo}")

        instancia = carregar_instancia(caminho)

        status_str, ub, lb, gap = resolver_modelo_exato(instancia, tempo_limite_segundos=tempo_limite)

        with open(arquivo_saida, "a") as f:
            if status_str != "SEM_SOLUCAO":
                linha = f"{nome_arquivo:<30} | {status_str:<15} | {ub:<15.2f} | {lb:<15.2f} | {gap:<10.2f}\n"
            else:
                linha = f"{nome_arquivo:<30} | {'SEM_SOLUCAO':<15} | {'-':<15} | {'-':<15} | {'-':<10}\n"
            f.write(linha)