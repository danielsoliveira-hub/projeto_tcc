"""
================================================================================
INTERFACE VISUAL PARA O MOTOR DE CRÉDITO (STREAMLIT)
================================================================================

OBJETIVO:
- Criar uma interface amigável para interação com a API de Crédito Responsável.
- Exibir resultados de forma clara (semáforo, decisão, justificativa).
"""

# ==============================================================================
# BLOCO 1: IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
import streamlit as st
import requests

# ==============================================================================
# BLOCO 2: CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Credit Score AI",
    page_icon="🏦",
    layout="centered"
)

# ==============================================================================
# BLOCO 3: CABEÇALHO E INTRODUÇÃO
# ==============================================================================
st.title("🏦 Sistema de Análise de Crédito")
st.markdown("---")
st.write("""
Preencha os dados do cliente abaixo para obter a recomendação de crédito em tempo real.
A análise combina **Inteligência Artificial** com as regras da **Lei 14.181**.
""")

# ==============================================================================
# BLOCO 4: FORMULÁRIO DE INPUT
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Dados Pessoais")
    age = st.number_input("Idade", min_value=18, max_value=100, value=35)
    dependents = st.number_input("Número de Dependentes", min_value=0, max_value=20, value=0)
    income = st.number_input("Renda Mensal (R$)", min_value=0.0, value=5000.0)

with col2:
    st.subheader("💰 Situação Financeira")
    # DebtRatio: usuário informa em %, convertemos para decimal
    debt_ratio_pct = st.slider("Comprometimento de Renda (%)", 0, 100, 30)
    debt_ratio = debt_ratio_pct / 100.0
    
    # Utilização do cartão: usuário informa em %, convertemos para decimal
    utilization_pct = st.slider("Uso do Limite do Cartão (%)", 0, 150, 10)
    utilization = utilization_pct / 100.0

st.markdown("---")
st.subheader("📜 Histórico de Crédito")

col3, col4, col5 = st.columns(3)
with col3:
    atraso_curto = st.number_input("Atrasos 30-59 dias", min_value=0, value=0)
    atraso_medio = st.number_input("Atrasos 60-89 dias", min_value=0, value=0)

with col4:
    atraso_grave = st.number_input("Atrasos > 90 dias", min_value=0, value=0)
    emprestimos = st.number_input("Linhas de Crédito Abertas", min_value=0, value=5)

with col5:
    imoveis = st.number_input("Financiamentos Imobiliários", min_value=0, value=0)

# ==============================================================================
# BLOCO 5: BOTÃO DE AÇÃO
# ==============================================================================
if st.button("🔍 ANALISAR CRÉDITO", use_container_width=True):
    
    # 1. Montar o JSON (payload enviado à API)
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": atraso_curto,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": emprestimos,
        "NumberOfTimes90DaysLate": atraso_grave,
        "NumberRealEstateLoansOrLines": imoveis,
        "NumberOfTime60_89DaysPastDueNotWorse": atraso_medio,
        "NumberOfDependents": dependents,
        "DebtRatio": debt_ratio
    }
    
    # 2. Enviar para a API
    try:
        with st.spinner('Consultando Inteligência Artificial...'):
            response = requests.post("http://127.0.0.1:8000/analisar_credito", json=payload)
        
        # 3. Exibir a Resposta
        if response.status_code == 200:
            resultado = response.json()
            
            # --- Busca inteligente de chaves (compatível com diferentes versões da API) ---
            decisao = resultado.get("decisao_final") or resultado.get("decisao")
            mensagem = resultado.get("justificativa") or resultado.get("mensagem") or resultado.get("detalhes")
            cor = resultado.get("semaforo")
            detalhes_tec = resultado.get("analise_tecnica") or resultado.get("detalhes_tecnicos") or resultado.get("scores")
            
            prob_calote = detalhes_tec.get("probabilidade_calote") or detalhes_tec.get("score_inadimplencia") or detalhes_tec.get("calote")
            prob_esg = detalhes_tec.get("probabilidade_superendividamento") or detalhes_tec.get("score_superendividamento") or detalhes_tec.get("esg")

            # ==============================================================================
            # BLOCO 6: RESULTADOS VISUAIS
            # ==============================================================================
            st.markdown("### Resultado da Análise:")
            
            if cor == "VERDE":
                st.success(f"✅ {decisao}")
                st.balloons()
            elif cor == "AMARELO":
                st.warning(f"⚠️ {decisao}")
            else:
                st.error(f"🚫 {decisao}")
            
            st.info(f"**Motivo:** {mensagem}")
            
            # Detalhes técnicos em seção expansível
            with st.expander("Ver Detalhes Técnicos (Probabilidades)"):
                st.write(f"**Risco de Inadimplência (Modelo A):** {prob_calote*100:.2f}%")
                st.write(f"**Risco de Superendividamento (Modelo B):** {prob_esg*100:.2f}%")
                
                # Termômetro de risco (barra de progresso)
                st.write("Termômetro de Risco Financeiro:")
                st.progress(prob_calote)
                
        else:
            st.error(f"Erro na API: {response.status_code}")
            st.write(response.text)
            
    except Exception as e:
        st.error(f"Erro de conexão ou processamento: {e}")