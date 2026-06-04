"""
database.py
-----------
Persistência das predições em banco SQLite (sem dependências externas).
Atende às funcionalidades adicionais: salvar predições e consultá-las.
"""

import os
import sqlite3
from datetime import datetime

_CAMINHO_DB = os.path.join(os.path.dirname(__file__), "instances", "predicoes.db")


def _conectar():
    conn = sqlite3.connect(_CAMINHO_DB)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar():
    """Cria a tabela de predições caso ainda não exista."""
    with _conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predicoes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT,
                age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER,
                chol INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER,
                exang INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER,
                predicao INTEGER, prob_com REAL
            )
        """)


def salvar(valores: dict, resultado: dict):
    """Salva uma predição realizada."""
    with _conectar() as conn:
        conn.execute("""
            INSERT INTO predicoes
            (data_hora, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
             exang, oldpeak, slope, ca, thal, predicao, prob_com)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().strftime("%d/%m/%Y %H:%M"),
            valores["age"], valores["sex"], valores["cp"], valores["trestbps"],
            valores["chol"], valores["fbs"], valores["restecg"], valores["thalach"],
            valores["exang"], valores["oldpeak"], valores["slope"], valores["ca"],
            valores["thal"], resultado["predicao"], resultado["prob_com"],
        ))


def listar(limite=100):
    """Retorna as predições mais recentes."""
    with _conectar() as conn:
        cur = conn.execute(
            "SELECT * FROM predicoes ORDER BY id DESC LIMIT ?", (limite,)
        )
        return [dict(r) for r in cur.fetchall()]


def estatisticas():
    """Resumo para o dashboard."""
    with _conectar() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predicoes").fetchone()[0]
        com = conn.execute(
            "SELECT COUNT(*) FROM predicoes WHERE predicao = 1").fetchone()[0]
        sem = total - com
    return {"total": total, "com": com, "sem": sem}
