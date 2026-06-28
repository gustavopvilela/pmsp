import copy
import math
import random
import time
import numpy as np
from utils import UPMSPInstance, calcular_tempo_maquina, calcular_tempos_maquinas

"""
Representação da solução:
solucao = [[0, 2], [1, 3, 4], [], []]
A máquina 0 executa os jobs 0 e 2, as máquinas 2 e 3 estão ociosas
"""

def gerar_solucao_inicial (instancia: UPMSPInstance):
    """
    Gera uma solução inicial aleatória distribuindo
    todos os jobs pelas máquinas disponíveis.
    """
    solucao = [[] for _ in range(instancia.maquinas)]

    jobs = list(range(instancia.jobs))
    random.shuffle(jobs)

    for job in jobs:
        maquina_escolhida = random.randint(0, instancia.maquinas - 1)
        solucao[maquina_escolhida].append(job)

    return solucao

def gerar_vizinho (solucao_atual, tempos_atuais, instancia: UPMSPInstance, taxa=0.7):
    """
    Gera um vizinho com viés heurístico.
    :param tempos_atuais: vetor de makespan de cada máquina
    :param instancia: classe UPMSPInstance da instância atual
    :param solucao_atual: solução atual do sistema
    :param taxa: 0.0 (busca cega total) até 1.0 (100% guloso focado em melhora)
    """
    vizinho = copy.deepcopy(solucao_atual)

    maquina_mais_demorada = int(np.argmax(tempos_atuais))
    maquina_menos_demorada = int(np.argmin(tempos_atuais))
    if not vizinho[maquina_mais_demorada]:
        return vizinho, set()

    tipo_movimento = random.choice(["shift", "swap"])
    usar_inteligencia = random.random() < taxa
    print(f"Tipo Movimento: {tipo_movimento} {"inteligente" if usar_inteligencia else "normal"}")
    maquinas_afetadas = set()

    if tipo_movimento == "shift":
        if usar_inteligencia and maquina_mais_demorada != maquina_menos_demorada:
            # Tira um job da máquina que mais tem e coloca na que menos tem
            maquina_origem = maquina_mais_demorada
            maquina_destino = maquina_menos_demorada
        else:
            # Shift aleatório
            maquinas_ocupadas = [m for m in range(instancia.maquinas) if vizinho[m]]
            maquina_origem = random.choice(maquinas_ocupadas)
            maquina_destino = random.randint(0, instancia.maquinas - 1)

        idx_job = random.randrange(len(vizinho[maquina_origem]))
        job = vizinho[maquina_origem].pop(idx_job)
        pos_destino = random.randint(0, len(vizinho[maquina_destino]))
        vizinho[maquina_destino].insert(pos_destino, job)
        maquinas_afetadas.add(maquina_origem)
        maquinas_afetadas.add(maquina_destino)

    elif tipo_movimento == "swap":
        maquinas_ocupadas = [m for m in range(instancia.maquinas) if vizinho[m]]

        if usar_inteligencia and len(maquinas_ocupadas) > 1:
            maquina1 = maquina_mais_demorada
            outras = [m for m in maquinas_ocupadas if m != maquina1]
            maquina2 = random.choice(outras) if outras else maquina1
        else:
            maquina1 = random.choice(maquinas_ocupadas)
            maquina2 = random.choice(maquinas_ocupadas)

        idx1 = random.randrange(len(vizinho[maquina1]))
        idx2 = random.randrange(len(vizinho[maquina2]))
        vizinho[maquina1][idx1], vizinho[maquina2][idx2] = vizinho[maquina2][idx2], vizinho[maquina1][idx1]
        maquinas_afetadas.add(maquina1)
        maquinas_afetadas.add(maquina2)

    return vizinho, maquinas_afetadas

def atualizar_tempos (tempos_atuais, vizinho, maquinas_afetadas, instancia: UPMSPInstance):
    """
    Recalcula os tempos através de delta evaluation.
    """
    tempos_vizinho = tempos_atuais.copy()
    for maquina in maquinas_afetadas:
        tempos_vizinho[maquina] = calcular_tempo_maquina(vizinho[maquina], maquina, instancia)
    return tempos_vizinho

def simulated_annealing (instancia: UPMSPInstance, temp_inicial=1000, taxa_resfriamento=0.99, iteracoes_por_temp=100, tempo_limite_segundos=None, max_iteracoes_sem_melhora=None):
    solucao_atual = gerar_solucao_inicial(instancia)
    tempos_atuais = calcular_tempos_maquinas(solucao_atual, instancia)
    makespan_atual = np.max(tempos_atuais)
    melhor_solucao = copy.deepcopy(solucao_atual)
    melhor_makespan = makespan_atual

    T = temp_inicial
    T_final = 1.0

    historico = {
        'iteracao': [],
        'temperatura': [],
        'makespan_atual': [],
        'melhor_makespan': [],
    }
    iteracao_global = 0
    iteracoes_sem_melhora = 0
    inicio = time.time()
    motivo_parada = "temperatura mínima atingida"
    parar_busca = False

    while T > T_final and not parar_busca:
        for _ in range(iteracoes_por_temp):
            iteracao_global += 1
            vizinho, maquinas_afetadas = gerar_vizinho(solucao_atual, tempos_atuais, instancia, taxa=0.7)
            if not maquinas_afetadas: # Movimento não foi possível
                continue

            melhorou = False
            tempos_vizinho = atualizar_tempos(tempos_atuais, vizinho, maquinas_afetadas, instancia)
            makespan_vizinho = np.max(tempos_vizinho)
            delta = makespan_vizinho - makespan_atual

            if delta < 0:
                solucao_atual = vizinho
                makespan_atual = makespan_vizinho
                tempos_atuais = tempos_vizinho

                if makespan_atual < melhor_makespan:
                    melhor_makespan = makespan_atual
                    melhor_solucao = copy.deepcopy(solucao_atual)
                    melhorou = True

            else:
                probabilidade = math.exp(-delta / T)
                if random.random() < probabilidade:
                    solucao_atual = vizinho
                    makespan_atual = makespan_vizinho
                    tempos_atuais = tempos_vizinho

            iteracoes_sem_melhora = 0 if melhorou else iteracoes_sem_melhora + 1
            historico['iteracao'].append(iteracao_global)
            historico['temperatura'].append(T)
            historico['makespan_atual'].append(makespan_atual)
            historico['melhor_makespan'].append(melhor_makespan)

            if tempo_limite_segundos is not None and (time.time() - inicio) >= tempo_limite_segundos:
                motivo_parada = f"tempo limite de {tempo_limite_segundos}s atingido"
                parar_busca = True
                break

            if max_iteracoes_sem_melhora is not None and iteracoes_sem_melhora >= max_iteracoes_sem_melhora:
                motivo_parada = f"sem melhora por {max_iteracoes_sem_melhora} iterações consecutivas"
                parar_busca = True
                break

        T = T * taxa_resfriamento

    print(f"Simulated Annealing finalizado ({motivo_parada}) --"
          f"{iteracao_global} iterações em {time.time() - inicio:.1f}s")
    return melhor_solucao, melhor_makespan, historico