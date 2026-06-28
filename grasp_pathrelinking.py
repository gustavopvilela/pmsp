import copy
import random
import numpy as np
from utils import UPMSPInstance, calcular_tempo_maquina, calcular_tempos_maquinas, calcular_makespan


def recalcular_makespan_delta(tempos_atuais, maquina_alterada, nova_sequencia, instancia: UPMSPInstance):
    """
    Recalcula apenas o tempo da máquina afetada e retorna o novo makespan,
    poupando processamento de matrizes inteiras.
    """
    novo_tempo = calcular_tempo_maquina(nova_sequencia, maquina_alterada, instancia)
    tempos_simulados = tempos_atuais.copy()
    tempos_simulados[maquina_alterada] = novo_tempo
    return np.max(tempos_simulados), tempos_simulados


def obter_maquina_do_job(solucao, job):
    """Função auxiliar para encontrar em qual máquina um job está."""
    for m, jobs in enumerate(solucao):
        if job in jobs:
            return m
    return -1


def construcao_gulosa(instancia: UPMSPInstance, alpha=0.3):
    solucao = [[] for _ in range(instancia.maquinas)]
    tempos = np.zeros(instancia.maquinas)

    jobs_nao_alocados = list(range(instancia.jobs))  # nosso conjunto candidato
    random.shuffle(jobs_nao_alocados)

    for job in jobs_nao_alocados:
        custos = []
        # avalia o custo de inserir cada job em todas as posições
        for m in range(instancia.maquinas):
            for pos in range(len(solucao[m]) + 1):
                sol_temp = copy.copy(solucao[m])
                sol_temp.insert(pos, job)

                mk_vizinho, _ = recalcular_makespan_delta(tempos, m, sol_temp, instancia)
                custos.append((m, pos, mk_vizinho))

        c_min = min(custos, key=lambda x: x[2])[2]
        c_max = max(custos, key=lambda x: x[2])[2]

        limite = c_min + alpha * (c_max - c_min) + 1e-5

        # rcl so com as melhores insercoes
        rcl = [(m, pos, mk) for m, pos, mk in custos if mk <= limite]

        if not rcl:
            rcl = [(m, pos, mk) for m, pos, mk in custos if mk == c_min]
        if not rcl:
            rcl = [custos[0]]

        m_escolhida, pos_escolhida, novo_mk = random.choice(rcl)

        solucao[m_escolhida].insert(pos_escolhida, job)
        tempos[m_escolhida] = calcular_tempo_maquina(solucao[m_escolhida], m_escolhida, instancia)

    return solucao


def busca_local_avancada(solucao, instancia: UPMSPInstance):
    sol_atual = copy.deepcopy(solucao)
    tempos = calcular_tempos_maquinas(sol_atual, instancia)
    mk_atual = np.max(tempos)
    melhorou = True

    while melhorou:
        melhorou = False
        maquina_gargalo = int(np.argmax(tempos))

        if not sol_atual[maquina_gargalo]:
            break

        for i, job in enumerate(sol_atual[maquina_gargalo]):
            # tenta melhorar fazendo movimento intramaquina
            for pos in range(len(sol_atual[maquina_gargalo])):
                if i == pos: continue

                sol_viz = copy.copy(sol_atual[maquina_gargalo])
                sol_viz.pop(i)
                sol_viz.insert(pos, job)

                mk_viz, novos_tempos = recalcular_makespan_delta(tempos, maquina_gargalo, sol_viz, instancia)

                if mk_viz < mk_atual:
                    sol_atual[maquina_gargalo] = sol_viz
                    tempos = novos_tempos
                    mk_atual = mk_viz
                    melhorou = True
                    break
            if melhorou: break

            # tenta melhorar fazendo movimento intermaquina
            for m in range(instancia.maquinas):
                if m == maquina_gargalo: continue

                for pos in range(len(sol_atual[m]) + 1):
                    # simula a remoção
                    seq_gargalo_temp = copy.copy(sol_atual[maquina_gargalo])
                    seq_gargalo_temp.pop(i)
                    tempo_gargalo_novo = calcular_tempo_maquina(seq_gargalo_temp, maquina_gargalo, instancia)

                    # simula inserção
                    seq_m_temp = copy.copy(sol_atual[m])
                    seq_m_temp.insert(pos, job)
                    tempo_m_novo = calcular_tempo_maquina(seq_m_temp, m, instancia)

                    # verifica novo makespan
                    tempos_temp = tempos.copy()
                    tempos_temp[maquina_gargalo] = tempo_gargalo_novo
                    tempos_temp[m] = tempo_m_novo
                    mk_viz = np.max(tempos_temp)

                    if mk_viz < mk_atual:
                        sol_atual[maquina_gargalo] = seq_gargalo_temp
                        sol_atual[m] = seq_m_temp
                        tempos = tempos_temp
                        mk_atual = mk_viz
                        melhorou = True
                        break
                if melhorou: break
            if melhorou: break

    return sol_atual


def path_relinking(xs, xt, instancia: UPMSPInstance):
    """
    Movimento: Mover um job que está no lugar 'errado' para sua máquina destino.
    """
    delta = []
    for j in range(instancia.jobs):
        m_s = obter_maquina_do_job(xs, j)
        m_t = obter_maquina_do_job(xt, j)
        if m_s != m_t:
            delta.append((j, m_t))

    melhor_sol_pr = copy.deepcopy(xs)
    melhor_mk_pr = calcular_makespan(xs, instancia)
    x_atual = copy.deepcopy(xs)

    while len(delta) > 0:
        melhor_movimento = None
        melhor_mk_vizinho = float('inf')
        melhor_vizinho = None

        for (j, m_t) in delta:
            m_atual = obter_maquina_do_job(x_atual, j)
            x_temp = copy.deepcopy(x_atual)
            x_temp[m_atual].remove(j)

            for pos in range(len(x_temp[m_t]) + 1):
                x_viz = copy.deepcopy(x_temp)
                x_viz[m_t].insert(pos, j)
                mk_viz = calcular_makespan(x_viz, instancia)

                if mk_viz < melhor_mk_vizinho:
                    melhor_mk_vizinho = mk_viz
                    melhor_vizinho = x_viz
                    melhor_movimento = (j, m_t)

        x_atual = copy.deepcopy(melhor_vizinho)
        delta.remove(melhor_movimento)

        if melhor_mk_vizinho < melhor_mk_pr:
            melhor_mk_pr = melhor_mk_vizinho
            melhor_sol_pr = copy.deepcopy(x_atual)

    return melhor_sol_pr


def calcular_distancia(sol1, sol2, jobs_totais):
    """Retorna o percentual de jobs que estão alocados em máquinas diferentes."""
    jobs_diferentes = 0
    for j in range(jobs_totais):
        m1 = obter_maquina_do_job(sol1, j)
        m2 = obter_maquina_do_job(sol2, j)
        if m1 != m2:
            jobs_diferentes += 1
    return jobs_diferentes / jobs_totais


def atualizar_pool(pool, solucao, makespan, instancia: UPMSPInstance, max_elite=10, min_diff=0.1):
    """
    min_diff = 0.1 significa que a solução precisa ter pelo menos 10%
    dos jobs em máquinas diferentes das soluções que já estão no pool.
    """
    # verifica se a solução é estruturalmente diferente de TODAS no pool
    for sol_p, mk_p in pool:
        diff = calcular_distancia(solucao, sol_p, instancia.jobs)
        if diff < min_diff:
            return

    if len(pool) < max_elite:
        pool.append((solucao, makespan))
        pool.sort(key=lambda x: x[1])
    else:
        if makespan < pool[-1][1]:
            pool[-1] = (solucao, makespan)
            pool.sort(key=lambda x: x[1])


def grasp_path_relinking(instancia: UPMSPInstance, max_iter=50, alpha=0.3, max_elite=10):
    pool = []  # pool inicia vazio
    best_sol = None
    best_makespan = float('inf')

    for i in range(1, max_iter + 1):
        print(f"Iteração GRASP {i}/{max_iter}...")

        # construcao gulosa e busca local
        x = construcao_gulosa(instancia, alpha)
        x = busca_local_avancada(x, instancia)
        mk_x = calcular_makespan(x, instancia)

        if i >= 2 and len(pool) > 0:
            y, mk_y = random.choice(pool)

            xp = path_relinking(x, y, instancia)
            mk_xp = calcular_makespan(xp, instancia)

            atualizar_pool(pool, xp, mk_xp, instancia, max_elite)

            if mk_xp < best_makespan:
                best_makespan = mk_xp
                best_sol = copy.deepcopy(xp)
                print(f"  -> Novo Melhor (pós-PR): {best_makespan:.2f}")
        else:
            atualizar_pool(pool, x, mk_x, instancia, max_elite)
            if mk_x < best_makespan:
                best_makespan = mk_x
                best_sol = copy.deepcopy(x)
                print(f"  -> Novo Melhor: {best_makespan:.2f}")

    return best_sol, best_makespan