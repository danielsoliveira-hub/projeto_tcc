# 🏦 Motor de Crédito Responsável (Lei 14.181)

Este projeto implementa um **motor de decisão de crédito** que combina **Inteligência Artificial** com **regras rígidas da Lei 14.181**, protegendo tanto o **Banco** (risco de inadimplência) quanto o **Cliente** (risco de superendividamento).

## 📌 Estrutura do Projeto

- **Treinamento (`projeto_final_credito.py`)**
  - Limpeza e preparação dos dados (`cs-training.csv`)
  - Engenharia de variáveis (`Renda_Per_Capita`, `Total_Atrasos`, etc.)
  - Treinamento de dois modelos Random Forest:
    - Modelo A → risco de inadimplência
    - Modelo B → risco de superendividamento
  - Salvamento dos artefatos em `.pkl`

- **API (`app.py`)**
  - Implementada em **FastAPI**
  - Endpoint `/analisar_credito` recebe dados do cliente e retorna:
    - Decisão final (APROVADO, NEGADO, ALERTA)
    - Semáforo (VERDE, AMARELO, VERMELHO)
    - Justificativa textual
    - Probabilidades técnicas (calote e superendividamento)

- **Interface (`interface.py`)**
  - Implementada em **Streamlit**
  - Formulário para entrada de dados do cliente
  - Conexão com a API
  - Exibição dos resultados com gráficos e indicadores visuais

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```
### 2. Treinar os modelos
````
python projeto_final_credito.py
````

### 3. Subir a API

uvicorn app:app --reload

### 4. Rodar a interface

streamlit run interface.py