import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def comparar_duas_metaheuristicas(gaps_m1, gaps_m2, nome_m1, nome_m2):
    # Realiza o Teste de Wilcoxon Signed-Rank
    stat, p_value = stats.wilcoxon(gaps_m1, gaps_m2)

    media_m1 = np.mean(gaps_m1)
    media_m2 = np.mean(gaps_m2)

    fig, ax = plt.subplots(figsize=(7, 4))

    texto = f"Comparação de GAPs:\n{nome_m1} vs {nome_m2}\n\n"
    texto += f"Gap Médio {nome_m1}: {media_m1:.2f}%\n"
    texto += f"Gap Médio {nome_m2}: {media_m2:.2f}%\n\n"
    texto += f"--- Teste de Wilcoxon ---\n"
    texto += f"Estatística (W): {stat:.2f}\n"
    texto += f"p-value: {p_value:.4e}\n\n"

    if p_value < 0.05:
        texto += "CONCLUSÃO: Diferença estatística significativa (p < 0.05)\n"
        melhor = nome_m1 if media_m1 < media_m2 else nome_m2
        texto += f"Vencedor: {melhor}"
        cor = "#d4edda"  # verde
    else:
        texto += "CONCLUSÃO: Não há diferença significativa (p >= 0.05)\n"
        texto += "Empate Técnico."
        cor = "#fff3cd"  # amarelo

    ax.text(0.5, 0.5, texto, ha='center', va='center', fontsize=11,
            bbox=dict(facecolor=cor, edgecolor='black', boxstyle='round,pad=1'))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig('analise_wilcoxon.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> analise_wilcoxon.png salvo.")