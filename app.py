"""
app.py
------
Arquivo principal da aplicação Flask. Contém as rotas e inicia o servidor.

Rotas:
  /            -> formulário das variáveis + resultado da predição
  /api/predict -> endpoint JSON que recebe os dados e retorna a predição
  /predicoes   -> consulta das predições já realizadas (dados preditos)
  /dashboard   -> estatísticas e gráfico das predições
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio

from classifier import prever, FEATURES, NOME_MODELO
import database

app = Flask(__name__)
database.inicializar()

# Metadados das variáveis: rótulo, tipo e opções (para montar o formulário)
CAMPOS = [
    {"nome": "age", "label": "Idade (anos)", "tipo": "number", "min": 20, "max": 100},
    {"nome": "sex", "label": "Sexo", "tipo": "select",
     "opcoes": [(1, "Masculino"), (0, "Feminino")]},
    {"nome": "cp", "label": "Tipo de dor no peito", "tipo": "select",
     "opcoes": [(0, "Angina típica"), (1, "Angina atípica"),
                (2, "Dor não anginosa"), (3, "Assintomático")]},
    {"nome": "trestbps", "label": "Pressão arterial em repouso (mmHg)",
     "tipo": "number", "min": 80, "max": 220},
    {"nome": "chol", "label": "Colesterol sérico (mg/dl)",
     "tipo": "number", "min": 100, "max": 600},
    {"nome": "fbs", "label": "Glicemia em jejum > 120 mg/dl", "tipo": "select",
     "opcoes": [(1, "Sim"), (0, "Não")]},
    {"nome": "restecg", "label": "Eletrocardiograma em repouso", "tipo": "select",
     "opcoes": [(0, "Normal"), (1, "Anormalidade ST-T"),
                (2, "Hipertrofia ventricular")]},
    {"nome": "thalach", "label": "Frequência cardíaca máxima",
     "tipo": "number", "min": 60, "max": 220},
    {"nome": "exang", "label": "Angina induzida por exercício", "tipo": "select",
     "opcoes": [(1, "Sim"), (0, "Não")]},
    {"nome": "oldpeak", "label": "Depressão de ST (oldpeak)",
     "tipo": "number", "step": "0.1", "min": 0, "max": 7},
    {"nome": "slope", "label": "Inclinação do segmento ST", "tipo": "select",
     "opcoes": [(0, "Ascendente"), (1, "Plano"), (2, "Descendente")]},
    {"nome": "ca", "label": "Nº de vasos principais coloridos", "tipo": "select",
     "opcoes": [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4")]},
    {"nome": "thal", "label": "Talassemia", "tipo": "select",
     "opcoes": [(1, "Normal"), (2, "Defeito fixo"), (3, "Defeito reversível"),
                (0, "Não informado")]},
]


def _extrair_valores(origem):
    """Converte os dados recebidos (form ou JSON) no dicionário esperado."""
    valores = {}
    for f in FEATURES:
        bruto = origem.get(f)
        if bruto is None or bruto == "":
            raise ValueError(f"Variável '{f}' não informada.")
        valores[f] = float(bruto) if f == "oldpeak" else int(float(bruto))
    return valores


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        try:
            valores = _extrair_valores(request.form)
            resultado = prever(valores)
            database.salvar(valores, resultado)
        except ValueError as e:
            resultado = {"erro": str(e)}
    return render_template(
        "index.html", campos=CAMPOS, resultado=resultado, modelo=NOME_MODELO
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON para integração. Recebe as variáveis e retorna a predição."""
    dados = request.get_json(silent=True) or {}
    try:
        valores = _extrair_valores(dados)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    resultado = prever(valores)
    database.salvar(valores, resultado)
    return jsonify(resultado)


@app.route("/predicoes")
def predicoes():
    """Consulta das predições já realizadas."""
    return render_template("predicoes.html", registros=database.listar())


@app.route("/dashboard")
def dashboard():
    stats = database.estatisticas()
    grafico = None
    if stats["total"] > 0:
        dados = pd.DataFrame({
            "Resultado": ["Sem doença", "Com doença"],
            "Quantidade": [stats["sem"], stats["com"]],
        })
        fig = px.bar(dados, x="Resultado", y="Quantidade", color="Resultado",
                     color_discrete_map={"Sem doença": "#2a9d8f",
                                         "Com doença": "#e76f51"},
                     title="Distribuição das predições realizadas")
        fig.update_layout(showlegend=False, margin=dict(t=50, b=20))
        grafico = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return render_template("dashboard.html", stats=stats, grafico=grafico)


if __name__ == "__main__":
    app.run(debug=True)
