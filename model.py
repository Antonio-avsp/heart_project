"""
model.py
--------
Responsável pelo treinamento, pré-processamento e geração do modelo preditivo.

Executa a Etapa 2 do projeto (experimentos):
  - Treina 5 modelos de classificação;
  - Roda cada modelo 30 vezes (splits diferentes) registrando média e desvio padrão;
  - Seleciona o melhor modelo com base no F1-Score médio;
  - Exporta o pipeline final (StandardScaler + modelo) em models/modelo_preditivo.pkl

Uso:  python model.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

CAMINHO_DADOS = "instances/heart.csv"
CAMINHO_MODELO = "models/modelo_preditivo.pkl"
N_EXECUCOES = 30  # exigência: no mínimo 30 execuções

# Ordem das variáveis explicativas usadas no formulário/predição
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
ALVO = "target"


def carregar_dados(caminho=CAMINHO_DADOS):
    """Carrega e faz a limpeza básica dos dados."""
    df = pd.read_csv(caminho)
    df = df.drop_duplicates().reset_index(drop=True)  # remove 1 duplicata
    return df


def definir_modelos():
    """Retorna os 5 modelos, cada um em um pipeline com escalonamento."""
    return {
        "Regressao Logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Arvore de Decisao": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier()),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
    }


def rodar_experimentos(df, n=N_EXECUCOES):
    """Treina cada modelo n vezes com splits diferentes e coleta métricas."""
    X = df[FEATURES].values
    y = df[ALVO].values

    registros = []
    for nome, modelo in definir_modelos().items():
        accs, precs, recs, f1s = [], [], [], []
        for i in range(n):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=i, stratify=y
            )
            modelo.fit(X_tr, y_tr)
            y_pred = modelo.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            precs.append(precision_score(y_te, y_pred, zero_division=0))
            recs.append(recall_score(y_te, y_pred, zero_division=0))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))

        registros.append({
            "Modelo": nome,
            "Acuracia_media": np.mean(accs),
            "Acuracia_std": np.std(accs),
            "Precisao_media": np.mean(precs),
            "Precisao_std": np.std(precs),
            "Recall_media": np.mean(recs),
            "Recall_std": np.std(recs),
            "F1_media": np.mean(f1s),
            "F1_std": np.std(f1s),
        })

    return pd.DataFrame(registros)


def exportar_melhor_modelo(df, nome_melhor, metricas=None):
    """Treina o melhor modelo no conjunto completo e o exporta com joblib."""
    X = df[FEATURES].values
    y = df[ALVO].values
    modelo = definir_modelos()[nome_melhor]
    modelo.fit(X, y)  # treino final em 100% dos dados

    pacote = {
        "pipeline": modelo,
        "features": FEATURES,
        "nome_modelo": nome_melhor,
        "metricas": metricas,
    }
    joblib.dump(pacote, CAMINHO_MODELO)
    return CAMINHO_MODELO


def main():
    df = carregar_dados()
    print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas\n")

    resultados = rodar_experimentos(df)

    # Tabela formatada (media +/- desvio)
    tabela = pd.DataFrame({
        "Modelo": resultados["Modelo"],
        "Acuracia": resultados.apply(lambda r: f"{r.Acuracia_media:.3f} +/- {r.Acuracia_std:.3f}", axis=1),
        "Precisao": resultados.apply(lambda r: f"{r.Precisao_media:.3f} +/- {r.Precisao_std:.3f}", axis=1),
        "Recall":   resultados.apply(lambda r: f"{r.Recall_media:.3f} +/- {r.Recall_std:.3f}", axis=1),
        "F1-Score": resultados.apply(lambda r: f"{r.F1_media:.3f} +/- {r.F1_std:.3f}", axis=1),
    })
    print("Resultados dos experimentos (media +/- desvio em 30 execucoes):\n")
    print(tabela.to_string(index=False))

    linha_melhor = resultados.loc[resultados["F1_media"].idxmax()]
    melhor = linha_melhor["Modelo"]
    print(f"\nMelhor modelo (maior F1 medio): {melhor}")

    # Métricas médias do melhor modelo (30 execuções), salvas junto ao .pkl
    # para serem exibidas na aplicação web.
    metricas = {
        "acuracia": round(float(linha_melhor["Acuracia_media"]) * 100, 1),
        "precisao": round(float(linha_melhor["Precisao_media"]) * 100, 1),
        "sensibilidade": round(float(linha_melhor["Recall_media"]) * 100, 1),
        "f1": round(float(linha_melhor["F1_media"]) * 100, 1),
    }

    caminho = exportar_melhor_modelo(df, melhor, metricas)
    print(f"Modelo exportado em: {caminho}")

    resultados.to_csv("instances/resultados_experimentos.csv", index=False)
    print("Tabela de resultados salva em: instances/resultados_experimentos.csv")


if __name__ == "__main__":
    main()
