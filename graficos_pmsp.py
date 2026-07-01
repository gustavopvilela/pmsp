import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd


def gerar_grafico_limites_vs_resultados(nomes_instancias, lb, ub, best, average, worst, nome_heuristica):
    """
    Gera um gráfico comparando os limites exatos (LB e UB) com os resultados
    (Best, Average, Worst) da meta-heurística por instância.
    """
    x = np.arange(len(nomes_instancias))

    nomes_limpos = [nome.replace('.txt', '').replace('_rep_', ' R') for nome in nomes_instancias]

    fig, ax = plt.subplots(figsize=(16, 7))

    #desenha a linha conectando LB e UB para mostrar o "range" do modelo exato
    ax.vlines(x, lb, ub, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Range Exato (LB - UB)', zorder=1)

    #plota os limitantes do CP-SAT (triângulos pretos)
    ax.scatter(x, ub, color='black', marker='v', s=70, label='Upper Bound (UB)', zorder=2)
    ax.scatter(x, lb, color='black', marker='^', s=70, label='Lower Bound (LB)', zorder=2)

    #plota os resultados da meta-heurística
    #best como estrela verde (destaque maior)
    ax.scatter(x, best, color='limegreen', marker='*', s=150, edgecolor='black', linewidth=0.5,
               label=f'Best ({nome_heuristica})', zorder=3)

    #average como bolinha azul
    ax.scatter(x, average, color='dodgerblue', marker='o', s=60, alpha=0.8, edgecolor='black', linewidth=0.5,
               label=f'Average ({nome_heuristica})', zorder=3)

    #worst como um 'X' vermelho
    ax.scatter(x, worst, color='crimson', marker='X', s=70, alpha=0.8, edgecolor='black', linewidth=0.5,
               label=f'Worst ({nome_heuristica})', zorder=3)

    #configurações de texto e eixos
    ax.set_ylabel('Valor da Solução (Makespan)', fontsize=11, fontweight='bold')
    ax.set_title(f'Desempenho da {nome_heuristica} vs. Limites Exatos (CP-SAT)', fontsize=14, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes_limpos, rotation=45, ha='right', fontsize=9)

    #legenda fora do gráfico para não cobrir os dados
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    #grid e layout
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    #salva o arquivo dinamicamente com o nome da heurística
    nome_arquivo = f'analise_limites_vs_{nome_heuristica.lower().replace(" ", "_")}.png'
    plt.savefig(nome_arquivo, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f" -> {nome_arquivo} salvo.")

def gerar_grafico_barras_runtime(runtime_m1, runtime_m2, nomes_instancias, nome_m1, nome_m2):
    x = np.arange(len(nomes_instancias))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - largura/2, runtime_m1, largura, label=nome_m1, color='dodgerblue')
    ax.bar(x + largura/2, runtime_m2, largura, label=nome_m2, color='crimson')

    ax.set_ylabel('Tempo (Segundos)')
    ax.set_title('Tempo de Execução por Instância', fontsize=14, fontweight='bold', pad=15)

    ax.set_xticks(x)
    #tira o ".txt" e outros sufixos gigantes para o texto caber no eixo X
    nomes_limpos = [nome.replace('.txt', '').replace('_rep_', ' R') for nome in nomes_instancias]
    ax.set_xticklabels(nomes_limpos, rotation=45, ha='right', fontsize=9)

    ax.legend()
    plt.tight_layout()
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('analise_barras_runtime.png', format='png', dpi=300)
    plt.close()
    print(" -> analise_barras_runtime.png salvo.")

def gerar_grafico_intervalo_confianca(gaps_m1, gaps_m2, nome_m1, nome_m2):
    """
    Gera um gráfico de barras com Intervalo de Confiança de 95% comparando duas meta-heurísticas
    recebendo diretamente as listas de gaps.
    """
    dados = [gaps_m1, gaps_m2]
    nomes_algoritmos = [nome_m1, nome_m2]

    medias = []
    margens_erro = []
    nivel_confianca = 0.95

    #cálculos Estatísticos
    for amostra in dados:
        n = len(amostra)
        media = np.mean(amostra)

        #erro padrão da média
        erro_padrao = stats.sem(amostra)

        #valor crítico T para 95% de confiança
        valor_critico_t = stats.t.ppf((1 + nivel_confianca) / 2.0, n - 1)

        #margem de erro
        margem = erro_padrao * valor_critico_t

        medias.append(media)
        margens_erro.append(margem)

    #construção do Gráfico
    fig, ax = plt.subplots(figsize=(9, 7))
    cores = ['dodgerblue', 'crimson']

    barras = ax.bar(nomes_algoritmos, medias, yerr=margens_erro, capsize=12,
                    color=cores, edgecolor='black', alpha=0.85, zorder=3,
                    error_kw={'elinewidth': 2, 'capthick': 2})

    ax.set_ylabel('Gap Médio em relação ao UB (%)', fontsize=12, fontweight='bold')
    ax.set_title('Desempenho Médio do GAP\n(Intervalo de Confiança 95%)', fontsize=14, fontweight='bold', pad=15)

    #adiciona a linha do zero
    ax.axhline(0, color='black', linewidth=1.5, zorder=4)

    #textos dinâmicos no meio da barra
    for barra, media in zip(barras, medias):
        altura = barra.get_height()

        #posição Y no meio da barra (mesmo se for negativa)
        pos_y = altura / 2

        sinal = "+" if media > 0 else ""
        texto = f'{sinal}{media:.1f}%'

        ax.text(barra.get_x() + barra.get_width() / 2, pos_y, texto,
                ha='center', va='center', color='white',
                fontweight='bold', fontsize=13, zorder=5,
                bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('analise_intervalo_confianca.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(" -> analise_intervalo_confianca.png salvo com sucesso.")

def gerar_ic_por_tamanho(instancias, gaps_m1, gaps_m2, nome_m1, nome_m2):
    """
    Gera um gráfico de Intervalo de Confiança agrupado pelo tamanho das instâncias.
    """
    #agrupar os GAPs pelo tamanho da instância (50, 100, 150, 200, 250)
    grupos_m1 = {}
    grupos_m2 = {}

    for inst, g1, g2 in zip(instancias, gaps_m1, gaps_m2):
        #extrai o número antes do 'x' (ex: "150x12_..." -> 150)
        tamanho = int(inst.split('x')[0])

        if tamanho not in grupos_m1:
            grupos_m1[tamanho] = []
            grupos_m2[tamanho] = []

        grupos_m1[tamanho].append(g1)
        grupos_m2[tamanho].append(g2)

    #ordenar os tamanhos para o eixo X ficar progressivo (50, 100, 150...)
    tamanhos_ordenados = sorted(grupos_m1.keys())
    labels_eixo_x = [f"{t} Tarefas" for t in tamanhos_ordenados]

    medias_m1, erros_m1 = [], []
    medias_m2, erros_m2 = [], []
    nivel_confianca = 0.95

    #calcular média e IC para cada grupo
    for tamanho in tamanhos_ordenados:
        para_m1 = grupos_m1[tamanho]
        para_m2 = grupos_m2[tamanho]

        n = len(para_m1)  #quantidade de amostras no grupo (geralmente 3: x4, x8, x12)

        if n > 1:
            t_critico = stats.t.ppf((1 + nivel_confianca) / 2.0, n - 1)
            erro_m1 = stats.sem(para_m1) * t_critico
            erro_m2 = stats.sem(para_m2) * t_critico
        else:
            erro_m1, erro_m2 = 0, 0  # Se houver só 1 instância, não há IC

        medias_m1.append(np.mean(para_m1))
        erros_m1.append(erro_m1)
        medias_m2.append(np.mean(para_m2))
        erros_m2.append(erro_m2)

    #construção do Gráfico Agrupado
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(tamanhos_ordenados))
    largura = 0.35

    barras_m1 = ax.bar(x - largura / 2, medias_m1, largura, yerr=erros_m1, label=nome_m1,
                       color='dodgerblue', edgecolor='black', capsize=8, alpha=0.85,
                       error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    barras_m2 = ax.bar(x + largura / 2, medias_m2, largura, yerr=erros_m2, label=nome_m2,
                       color='crimson', edgecolor='black', capsize=8, alpha=0.85,
                       error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.axhline(0, color='black', linewidth=1.5, zorder=0)

    #configurações visuais
    ax.set_ylabel('Gap Médio vs UB (%)', fontsize=12, fontweight='bold')
    ax.set_title('Intervalo de Confiança (95%) por Tamanho de Instância', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_eixo_x, fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=11)

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('analise_ic_agrupado.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(" -> analise_ic_agrupado.png salvo com sucesso.")