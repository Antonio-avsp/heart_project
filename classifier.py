"""
classifier.py
-------------
Contém a lógica de predição: carrega o modelo treinado e faz as predições.
O modelo é carregado uma única vez quando o módulo é importado.
"""

import os
import joblib
import numpy as np

_CAMINHO = os.path.join(os.path.dirname(__file__), "models", "modelo_preditivo.pkl")
_pacote = joblib.load(_CAMINHO)

_pipeline = _pacote["pipeline"]
FEATURES = _pacote["features"]
NOME_MODELO = _pacote["nome_modelo"]


def prever(valores: dict) -> dict:
    """
    Recebe um dicionário {nome_variavel: valor} e retorna a predição.

    Retorna:
        predicao (int): 0 = sem doença, 1 = com doença
        rotulo (str): descrição textual
        prob_sem / prob_com (float): probabilidades de cada classe
    """
    x = np.array([[float(valores[f]) for f in FEATURES]])
    predicao = int(_pipeline.predict(x)[0])
    prob = _pipeline.predict_proba(x)[0]

    return {
        "predicao": predicao,
        "rotulo": "Indício de doença cardíaca" if predicao == 1
                  else "Sem indício de doença cardíaca",
        "prob_sem": round(float(prob[0]) * 100, 1),
        "prob_com": round(float(prob[1]) * 100, 1),
    }
