import random
import numpy as np
from utils import UPMSPInstance, calcular_tempo_maquina, calcular_tempos_maquinas, calcular_makespan


def calcular_delta_insercao(job, m, pos, seq_m, instancia: UPMSPInstance):
    """Calcula em O(1) a variação de tempo ao inserir 'job' na máquina 'm' na posição 'pos'."""
    P = instancia.matriz_processamento
    S = instancia.matriz_setup

    delta = P[job][m]
    n = len(seq_m)

    if n == 0:
        return delta

    if pos == 0:
        return delta + S[m][job][seq_m[0]]

    if pos == n:
        return delta + S[m][seq_m[-1]][job]

    # Inserção no meio de dois jobs
    A = seq_m[pos - 1]
    B = seq_m[pos]
    delta += -S[m][A][B] + S[m][A][job] + S[m][job][B]

    return delta


def calcular_delta_remocao(pos, seq_m, m, instancia: UPMSPInstance):
    """Calcula em O(1) a variação de tempo ao remover o job da posição 'pos' da máquina 'm'."""
    P = instancia.matriz_processamento
    S = instancia.matriz_setup
    n = len(seq_m)
    job = seq_m[pos]

    delta = -P[job][m]

    if n == 1:
        return delta

    if pos == 0:
        return delta - S[m][job][seq_m[1]]

    if pos == n - 1:
        return delta - S[m][seq_m[-2]][job]

    # Remoção do meio de dois jobs
    A = seq_m[pos - 1]
    B = seq_m[pos + 1]
    delta += -S[m][A][job] - S[m][job][B] + S[m][A][B]

    return delta


def criar_mapa_jobs(solucao, num_jobs):
    """Cria um array O(1) para consultar em qual máquina o job está alocado."""
    mapa = np.zeros(num_jobs, dtype=int)
    for m, jobs in enumerate(solucao):
        for job in jobs:
            mapa[job] = m
    return mapa


def construcao_gulosa(instancia: UPMSPInstance, alpha=0.3):
    solucao = [[] for _ in range(instancia.maquinas)]
    tempos = np.zeros(instancia.maquinas)

    jobs_nao_alocados = list(range(instancia.jobs)) # nosso conjunto candidato
    random.shuffle(jobs_nao_alocados)

    for job in jobs_nao_alocados:
        custos = []
        # avalia o custo de inserir cada job em todas as posições
        for m in range(instancia.maquinas):
            for pos in range(len(solucao[m]) + 1):
                aumento = calcular_delta_insercao(job, m, pos, solucao[m], instancia)

                # Simula o makespan localmente
                tempos_simulados = tempos.copy()
                tempos_simulados[m] += aumento
                mk_vizinho = np.max(tempos_simulados)

                custos.append((m, pos, mk_vizinho, aumento))

        c_min = min(custos, key=lambda x: x[2])[2]
        c_max = max(custos, key=lambda x: x[2])[2]
        limite = c_min + alpha * (c_max - c_min) + 1e-5

        rcl = [(m, pos, mk, aum) for m, pos, mk, aum in custos if mk <= limite]
        if not rcl:
            rcl = [custos[0]]

        m_escolhida, pos_escolhida, novo_mk, aumento_escolhido = random.choice(rcl)

        solucao[m_escolhida].insert(pos_escolhida, job)
        tempos[m_escolhida] += aumento_escolhido

    return solucao


def busca_local_avancada(solucao, instancia: UPMSPInstance):
    sol_atual = [lista[:] for lista in solucao]
    tempos = calcular_tempos_maquinas(sol_atual, instancia)
    mk_atual = np.max(tempos)
    melhorou = True

    while melhorou:
        melhorou = False
        maquina_gargalo = int(np.argmax(tempos))

        if not sol_atual[maquina_gargalo]:
            break

        seq_gargalo = sol_atual[maquina_gargalo]

        for i, job in enumerate(seq_gargalo):
            # Shift (Intramaquina)
            for pos in range(len(seq_gargalo)):
                if i == pos: continue

                sol_viz = seq_gargalo[:]
                sol_viz.pop(i)
                sol_viz.insert(pos, job)

                tempo_novo = calcular_tempo_maquina(sol_viz, maquina_gargalo, instancia)
                if tempo_novo < tempos[maquina_gargalo]:
                    tempos_temp = tempos.copy()
                    tempos_temp[maquina_gargalo] = tempo_novo
                    mk_viz = np.max(tempos_temp)

                    if mk_viz < mk_atual:
                        sol_atual[maquina_gargalo] = sol_viz
                        tempos = tempos_temp
                        mk_atual = mk_viz
                        melhorou = True
                        break
            if melhorou: break

            # Relocate (Intermáquina)
            delta_rem = calcular_delta_remocao(i, seq_gargalo, maquina_gargalo, instancia)

            for m in range(instancia.maquinas):
                if m == maquina_gargalo: continue

                for pos in range(len(sol_atual[m]) + 1):
                    delta_ins = calcular_delta_insercao(job, m, pos, sol_atual[m], instancia)

                    tempos_temp = tempos.copy()
                    tempos_temp[maquina_gargalo] += delta_rem
                    tempos_temp[m] += delta_ins

                    mk_viz = np.max(tempos_temp)

                    if mk_viz < mk_atual:
                        # efetiva a alteração
                        sol_atual[maquina_gargalo].pop(i)
                        sol_atual[m].insert(pos, job)
                        tempos = tempos_temp
                        mk_atual = mk_viz
                        melhorou = True
                        break
                if melhorou: break
            if melhorou: break

    return sol_atual


def path_relinking(xs, xt, instancia: UPMSPInstance):
    mapa_s = criar_mapa_jobs(xs, instancia.jobs)
    mapa_t = criar_mapa_jobs(xt, instancia.jobs)

    delta = [(j, mapa_t[j]) for j in range(instancia.jobs) if mapa_s[j] != mapa_t[j]]

    melhor_sol_pr = [lista[:] for lista in xs]
    x_atual = [lista[:] for lista in xs]

    tempos_atuais = calcular_tempos_maquinas(x_atual, instancia)
    melhor_mk_pr = np.max(tempos_atuais)

    while len(delta) > 0:
        melhor_movimento = None
        melhor_mk_vizinho = float('inf')
        melhor_tempos_viz = None
        melhor_detalhes = None

        for (j, m_t) in delta:
            m_s = mapa_s[j]
            pos_j = x_atual[m_s].index(j)

            delta_rem = calcular_delta_remocao(pos_j, x_atual[m_s], m_s, instancia)

            for pos in range(len(x_atual[m_t]) + 1):
                delta_ins = calcular_delta_insercao(j, m_t, pos, x_atual[m_t], instancia)

                tempos_sim = tempos_atuais.copy()
                tempos_sim[m_s] += delta_rem
                tempos_sim[m_t] += delta_ins

                mk_viz = np.max(tempos_sim)

                if mk_viz < melhor_mk_vizinho:
                    melhor_mk_vizinho = mk_viz
                    melhor_movimento = (j, m_t)
                    melhor_detalhes = (m_s, pos_j, pos)
                    melhor_tempos_viz = tempos_sim

        # efetiva o melhor movimento da iteracao
        j, m_t = melhor_movimento
        m_s, pos_j, pos_ins = melhor_detalhes

        x_atual[m_s].pop(pos_j)
        x_atual[m_t].insert(pos_ins, j)
        tempos_atuais = melhor_tempos_viz
        mapa_s[j] = m_t

        delta.remove(melhor_movimento)

        if melhor_mk_vizinho < melhor_mk_pr:
            melhor_mk_pr = melhor_mk_vizinho
            melhor_sol_pr = [lista[:] for lista in x_atual]

    return melhor_sol_pr


def calcular_distancia(sol1, sol2, jobs_totais):
    """Retorna o percentual de jobs que estão alocados em máquinas diferentes."""
    mapa1 = criar_mapa_jobs(sol1, jobs_totais)
    mapa2 = criar_mapa_jobs(sol2, jobs_totais)
    jobs_diferentes = np.sum(mapa1 != mapa2)
    return jobs_diferentes / jobs_totais


def atualizar_pool(pool, solucao, makespan, instancia: UPMSPInstance, max_elite=10, min_diff=0.1):
    """
       min_diff = 0.1 significa que a solução precisa ter pelo menos 10%
       dos jobs em máquinas diferentes das soluções que já estão no pool.
    """
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
    pool = []
    best_sol = None
    best_makespan = float('inf')

    for i in range(1, max_iter + 1):
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
                best_sol = [lista[:] for lista in xp]
        else:
            atualizar_pool(pool, x, mk_x, instancia, max_elite)
            if mk_x < best_makespan:
                best_makespan = mk_x
                best_sol = [lista[:] for lista in x]

    return best_sol, best_makespan