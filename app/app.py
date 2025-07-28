import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento dos dados
@st.cache_data
def load_data():
    return pd.read_csv(r"data\datatran2025.csv", encoding="ISO-8859-1", delimiter=';')
    #return pd.read_csv(r"data\datatran2025_tratado.csv", delimiter=',')

df = load_data()
st.title("Análise de Regressão - Acidentes de Trânsito")

st.subheader("Prévia do Dataset")
st.dataframe(df.head())

# Seleção de variáveis
st.sidebar.header("Configurações do Modelo")
target = st.sidebar.selectbox("Variável alvo (Y)", df.select_dtypes(include=np.number).columns)

# Evita erro se target ainda não for selecionado
if target:
    feature_candidates = df.select_dtypes(include=np.number).columns.drop(target)
else:
    feature_candidates = df.select_dtypes(include=np.number).columns

features = st.sidebar.multiselect("Variáveis preditoras (X)", df.select_dtypes(include=np.number).columns.drop(target))

numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) == 0:
    st.error("O dataset não possui colunas numéricas disponíveis para regressão.")
    st.stop()


if features:
    X = df[features].dropna()
    y = df.loc[X.index, target]
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo sklearn
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Resultados da Regressão")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Plot real vs previsto
    st.subheader("Real vs Previsto")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    st.pyplot(fig)

    # OLS com Statsmodels
    st.subheader("📑 Sumário Estatístico (OLS)")
    X_ols = sm.add_constant(X)
    model_ols = sm.OLS(y, X_ols).fit()

    # Tabela de coeficientes
    summary_df = pd.DataFrame({
        "Coeficiente": model_ols.params,
        "Erro Padrão": model_ols.bse,
        "t-valor": model_ols.tvalues,
        "P-valor": model_ols.pvalues,
        "IC 2.5%": model_ols.conf_int()[0],
        "IC 97.5%": model_ols.conf_int()[1],
    })

    # Formatar e exibir a tabela
    st.markdown("#### 📊 Coeficientes do Modelo")
    st.dataframe(summary_df.style.format(precision=4))

    # Métricas gerais
    st.markdown("#### 📈 Métricas Gerais do Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{model_ols.rsquared:.3f}")
    col2.metric("R² Ajustado", f"{model_ols.rsquared_adj:.3f}")
    col3.metric("F-statistic", f"{model_ols.fvalue:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("AIC", f"{model_ols.aic:.2f}")
    col5.metric("BIC", f"{model_ols.bic:.2f}")
    col6.metric("Prob(F)", f"{model_ols.f_pvalue:.2e}")

else:
    st.warning("Selecione pelo menos uma variável preditora.")
