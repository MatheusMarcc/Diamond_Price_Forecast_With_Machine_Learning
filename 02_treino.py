import matplotlib.pyplot as plt
import numpy as np
from src import dados as D
from src.modelos import RegressaoLinearFechada, GradienteDrescedente

def main():
    # Carregando os dados simples
    conjunto = D.preparar(codificacao="ordinal", alvo_em_log=False, semente=42)
    X_treino, y_treino = conjunto["X_treino"], conjunto["y_treino"]
    colunas = conjunto["colunas"]

    # ========================================================
    # 1. ANÁLISE DOS PESOS
    # ========================================================
    print("Gerando gráfico de pesos...")
    modelo_fechado = RegressaoLinearFechada().treinar(X_treino, y_treino)
    pesos = modelo_fechado.coef_
    
    plt.figure(figsize=(10, 6))
    plt.barh(colunas, pesos, color=["#B4553B" if p < 0 else "#2F6690" for p in pesos])
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Pesos Atribuídos a Cada Característica", fontweight="bold")
    plt.xlabel("Valor do Peso")
    plt.tight_layout()
    plt.show()

    # ========================================================
    # 2. CONVERGÊNCIA (Onde os parâmetros param de mudar?)
    # ========================================================
    print("Gerando gráfico de convergência do Gradiente Descendente...")
    gd = GradienteDrescedente(taxa=0.1, epocas=1000).treinar(X_treino, y_treino)
    
    plt.figure(figsize=(8, 5))
    plt.plot(gd.historico_["custo"], color="#2F6690", linewidth=2)
    plt.title("Variação do Erro (MSE) por Época", fontweight="bold")
    plt.xlabel("Épocas (Iterações)")
    plt.ylabel("Custo (Erro)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    print(f"Os pesos pararam de sofrer grandes alterações por volta da época {gd.epocas_executadas_}.")

    # ========================================================
    # 3. EFEITO DA TAXA DE APRENDIZADO
    # ========================================================
    print("Gerando comparação de Taxas de Aprendizado...")
    taxas_para_testar = [0.001, 0.05, 0.1]
    cores = ["#B4553B", "#1F918B", "#2F6690"]
    
    plt.figure(figsize=(8, 5))
    for taxa, cor in zip(taxas_para_testar, cores):
        try:
            modelo = GradienteDrescedente(taxa=taxa, epocas=500).treinar(X_treino, y_treino)
            plt.plot(modelo.historico_["custo"], color=cor, label=f"Taxa = {taxa}")
        except:
            print(f"A taxa {taxa} fez o modelo explodir (divergiu)!")
            
    plt.title("Efeito de Diferentes Taxas de Aprendizado", fontweight="bold")
    plt.xlabel("Épocas (Iterações)")
    plt.ylabel("Custo (Erro)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()