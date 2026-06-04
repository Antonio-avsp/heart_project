# Sistema de Predição de Doença Cardíaca

Aplicação completa de Machine Learning que prevê o indício de doença cardíaca a
partir de variáveis clínicas, cobrindo desde o treinamento do modelo até a
disponibilização em uma aplicação web (Flask).

- **Problema:** Classificação binária (com / sem indício de doença cardíaca)
- **Base de dados:** UCI Heart Disease — Cleveland (303 registros, 13 variáveis)
- **Melhor modelo:** Regressão Logística (F1 ≈ 0,857 em 30 execuções)

## Estrutura do projeto

```
heart_project/
├── instances/
│   └── heart.csv                 # Base de dados utilizada
├── models/
│   └── modelo_preditivo.pkl      # Modelo treinado (pipeline + scaler)
├── static/
│   └── style.css                 # Estilos da aplicação
├── templates/
│   ├── base.html                 # Layout base
│   ├── index.html                # Formulário + resultado da predição
│   ├── dashboard.html            # Dashboard com estatísticas
│   └── predicoes.html            # Histórico das predições
├── app.py                        # Aplicação Flask (rotas)
├── classifier.py                 # Carrega o modelo e faz a predição
├── database.py                   # Persistência das predições (SQLite)
├── model.py                      # Treinamento e geração do .pkl
├── Relatorio_Experimentos.ipynb  # Notebook do Colab (Etapas 1 e 2)
└── requirements.txt              # Dependências
```

## Como executar

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. (Opcional) Treine o modelo novamente — gera `models/modelo_preditivo.pkl`:

   ```bash
   python model.py
   ```

3. Inicie a aplicação:

   ```bash
   python app.py
   ```

4. Acesse no navegador: <http://127.0.0.1:5000>

## Rotas disponíveis

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET/POST | Formulário das variáveis e resultado da predição |
| `/api/predict` | POST | Endpoint JSON que recebe os dados e retorna a predição |
| `/predicoes` | GET | Consulta das predições já realizadas |
| `/dashboard` | GET | Estatísticas e gráfico das predições |

### Exemplo de chamada à API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
       "slope":0,"ca":0,"thal":1}'
```

## Funcionalidades adicionais implementadas

- Persistência das predições em banco de dados (SQLite)
- Dashboard com estatísticas e gráfico das predições
- Endpoint de API (JSON) para integração

> Aviso: aplicação com finalidade educacional. Não substitui avaliação médica.
