import matplotlib.pyplot as plt
import numpy as np

def gerar_boxplot_gaps(gaps_m1, gaps_m2, nome_m1, nome_m2):
    dados_completos = [gaps_m1, gaps_m2]

    fig, ax = plt.subplots(figsize=(10, 8))

    bplot = ax.boxplot(dados_completos, patch_artist=True, tick_labels=[nome_m1, nome_m2])

    cores = ['lightblue', 'lightgreen']
    for caixa, cor in zip(bplot['boxes'], cores):
        caixa.set_facecolor(cor)

    ax.set_title(f'Distribuição dos Gaps em Relação ao Ótimo (CP-SAT)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Gap (%)')
    ax.set_xlabel('Metaheurísticas')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('analise_boxplot_gaps.png', format='png', dpi=300)
    plt.close() # Fecha a figura para não travar a execução
    print(" -> analise_boxplot_gaps.png salvo.")

def gerar_grafico_barras_runtime(runtime_m1, runtime_m2, nomes_instancias, nome_m1, nome_m2):
    x = np.arange(len(nomes_instancias))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - largura/2, runtime_m1, largura, label=nome_m1, color='dodgerblue')
    ax.bar(x + largura/2, runtime_m2, largura, label=nome_m2, color='crimson')

    ax.set_ylabel('Tempo (Segundos)')
    ax.set_title('Tempo de Execução por Instância', fontsize=14, fontweight='bold', pad=15)

    ax.set_xticks(x)
    # Tira o ".txt" e outros sufixos gigantes para o texto caber no eixo X
    nomes_limpos = [nome.replace('.txt', '').replace('_rep_', ' R') for nome in nomes_instancias]
    ax.set_xticklabels(nomes_limpos, rotation=45, ha='right', fontsize=9)

    ax.legend()
    plt.tight_layout()
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig('analise_barras_runtime.png', format='png', dpi=300)
    plt.close()
    print(" -> analise_barras_runtime.png salvo.")