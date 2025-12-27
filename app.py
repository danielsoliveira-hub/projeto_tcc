"""
================================================================================
API DO PROJETO FINAL: MOTOR DE CRÉDITO RESPONSÁVEL
================================================================================

OBJETIVO:
- Disponibilizar via FastAPI o motor de decisão de crédito responsável.
- Combinar Inteligência Artificial (modelos treinados) com regras rígidas da Lei 14.181.
- Garantir proteção tanto ao Banco (risco de inadimplência) quanto ao Cliente (risco de superendividamento).
"""

# ==============================================================================
# BLOCO 1: IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ==============================================================================
# BLOCO 2: VERIFICAÇÃO DE SEGURANÇA
# ==============================================================================
if not os.path.exists('modelo_inadimplencia.pkl'):
    print("ERRO: Arquivos .pkl não encontrados!")

# ==============================================================================
# BLOCO 3: CARGA DOS MODELOS E ARTEFATOS
# ==============================================================================
try:
    modelo_calote = joblib.load('modelo_inadimplencia.pkl')
    modelo_esg = joblib.load('modelo_superendividamento.pkl')
    artefatos = joblib.load('artefatos_projeto.pkl')
    print("API Iniciada: Modelos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    modelo_calote = None
    modelo_esg = None
    artefatos = None

# ==============================================================================
# BLOCO 4: DEFINIÇÃO DA API
# ==============================================================================
app = FastAPI(
    title="API Crédito Responsável",
    description="Motor de decisão Lei 14.181 (Proteção Banco + Cliente)",
    version="1.2"
)

# ==============================================================================
# BLOCO 5: SCHEMA DE DADOS
# ==============================================================================
# Usando Pydantic para validar os dados recebidos no endpoint.
# Cada campo corresponde às variáveis usadas no treinamento dos modelos.
class DadosCliente(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    MonthlyIncome: float = None
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float = None
    DebtRatio: float

# ==============================================================================
# BLOCO 6: ENDPOINT PRINCIPAL
# ==============================================================================
@app.post("/analisar_credito")
def analisar_credito(dados: DadosCliente):
    """
    Endpoint que recebe os dados de um cliente e retorna:
    - Decisão final (APROVADO, NEGADO, ALERTA)
    - Semáforo (VERDE, AMARELO, VERMELHO)
    - Justificativa textual
    - Análise técnica (probabilidades e dívida declarada)
    """

    # Caso os modelos não tenham sido carregados corretamente
    if not artefatos:
        return {"erro": "Modelos não carregados. Verifique o servidor."}

    # Converte os dados recebidos em dicionário
    input_dict = dados.dict()

    # --------------------------------------------------------------------------
    # PRÉ-PROCESSAMENTO DOS DADOS
    # --------------------------------------------------------------------------
    # Tratamento da renda: imputação da mediana se não informada
    if input_dict['MonthlyIncome'] is None or input_dict['MonthlyIncome'] == 0:
        renda = artefatos['valores_imputacao']['MonthlyIncome']
    else:
        renda = input_dict['MonthlyIncome']

    # Dependentes: assume 0 se não informado
    depend = input_dict['NumberOfDependents'] if input_dict['NumberOfDependents'] is not None else 0

    # Renda per capita: divide pela quantidade de pessoas (dependentes + titular)
    renda_per_capita = renda / (depend + 1)

    # Total de atrasos: soma dos diferentes tipos de atraso
    total_atrasos = (
        input_dict['NumberOfTime30_59DaysPastDueNotWorse'] +
        input_dict['NumberOfTime60_89DaysPastDueNotWorse'] +
        input_dict['NumberOfTimes90DaysLate']
    )

    # Uso crítico do cartão: flag binária (>=100% do limite)
    uso_critico = 1 if input_dict['RevolvingUtilizationOfUnsecuredLines'] >= 1.0 else 0

    # Montagem do vetor de features
    features_values = [
        input_dict['RevolvingUtilizationOfUnsecuredLines'],
        input_dict['age'],
        input_dict['NumberOfTime30_59DaysPastDueNotWorse'],
        renda,
        input_dict['NumberOfOpenCreditLinesAndLoans'],
        input_dict['NumberOfTimes90DaysLate'],
        input_dict['NumberRealEstateLoansOrLines'],
        input_dict['NumberOfTime60_89DaysPastDueNotWorse'],
        depend,
        renda_per_capita,
        total_atrasos,
        uso_critico
    ]

    features_array = np.array(features_values).reshape(1, -1)

    # --------------------------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------------------------
    # Modelo A (Calote): inclui DebtRatio_Clean
    debt_clean = 2.0 if input_dict['DebtRatio'] > 2.0 else input_dict['DebtRatio']
    feat_A = np.append(features_array, [[debt_clean]], axis=1)

    # Modelo B (Perfil ESG): não inclui DebtRatio
    feat_B = features_array

    # Probabilidades calculadas pelos modelos
    prob_calote = float(modelo_calote.predict_proba(feat_A)[0][1])
    prob_esg = float(modelo_esg.predict_proba(feat_B)[0][1])

    # --------------------------------------------------------------------------
    # MOTOR DE DECISÃO HÍBRIDO
    # --------------------------------------------------------------------------
    cortes = artefatos['limites']

    decisao = "APROVADO"
    cor = "VERDE"
    mensagem = "Cliente saudável."

    # PRIORIDADE 1: Trava de segurança (Lei 14.181)
    if input_dict['DebtRatio'] > cortes['vermelho']:
        decisao = "ALERTA (DADO DECLARADO)"
        cor = "AMARELO"
        mensagem = (
            f"O cliente declarou comprometimento de {input_dict['DebtRatio']*100:.0f}% da renda. "
            "Violação direta da Lei 14.181."
        )

    # PRIORIDADE 2: Proteção do Banco (risco de calote)
    elif prob_calote > cortes['corte_calote']:
        decisao = "NEGADO (RISCO CRÉDITO)"
        cor = "VERMELHO"
        mensagem = "Alto risco de inadimplência detectado pelo modelo histórico."

    # PRIORIDADE 3: Proteção do Cliente (risco de superendividamento futuro)
    elif prob_esg > cortes['corte_esg']:
        decisao = "ALERTA (PERFIL DE RISCO)"
        cor = "AMARELO"
        mensagem = "O comportamento financeiro indica alto risco de superendividamento futuro."

    # --------------------------------------------------------------------------
    # RETORNO DA API
    # --------------------------------------------------------------------------
    return {
        "decisao_final": decisao,
        "semaforo": cor,
        "justificativa": mensagem,
        "analise_tecnica": {
            "probabilidade_calote": round(prob_calote, 4),
            "probabilidade_superendividamento": round(prob_esg, 4),
            "divida_declarada": input_dict['DebtRatio']
        }
    }