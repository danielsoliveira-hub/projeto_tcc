**Aluno: Daniel Silva Oliveira**


# Relatório Acadêmico – Motor de Crédito Responsável

## 1. Business Understanding (Entendimento do Negócio)

### 1.1 Objetivos
- Banco: reduzir risco de inadimplência e minimizar a exposição ao risco legal.
- Cliente: prevenir superendividamento, evitando que o cliente assuma compromissos que extrapolem o seu orçamento e comprometam sua saúde financeira e subsistência (Lei 14.181).
- Critério de sucesso: modelos com AUC ≥ 0.70 e regras claras de decisão (semáforo verde, amarelo, vermelho).

### 1.2 Recursos
- Dados: cs-training.csv.
- Software: Python, Pandas, Scikit-Learn, Seaborn, Matplotlib, FastAPI, Streamlit.
- Infraestrutura: execução local com persistência em .pkl.

### 1.3 Perguntas de Negócio
- Qual a probabilidade de um cliente se tornar inadimplente?
- Qual a probabilidade de um cliente se tornar superendividado?
- Como aplicar a Lei 14.181 no processo de decisão de crédito?

## 2. Data Understanding (Entendimento dos Dados)

### 2.1 Coleta inicial
````
import pandas as pd
df = pd.read_csv("cs-training.csv")
print(df.shape)
df.head()
````
### 2.2 Descrição dos dados
````
df.info()
df.describe()
````
### 2.3 Qualidade dos dados
````
df.isnull().sum()
````
### 2.4 Outliers
````
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['DebtRatio'])
plt.title("Boxplot de DebtRatio")
plt.show()

sns.scatterplot(x=df['MonthlyIncome'], y=df['DebtRatio'])
plt.title("Dispersão: Renda vs DebtRatio")
plt.show()
````
![alt text](Figure_2-1.png)
![alt text](Figure_1.png)


## 3. Data Preparation (Preparação dos Dados)

### 3.1 Limpeza
 ````
 mediana_renda = df['MonthlyIncome'].median()
 df['MonthlyIncome'].fillna(mediana_renda, inplace=True)
 df['NumberOfDependents'].fillna(0, inplace=True)
 df['DebtRatio_Clean'] = df['DebtRatio'].apply(lambda x: 2.0 if x > 2.0 else x)
````
### 3.2 Feature Engineering
````
df.rename(columns={'SeriousDlqin2yrs':'Target_Inadimplencia'}, inplace=True)
df['Target_Superendividado'] = df['DebtRatio_Clean'].apply(lambda x: 1 if x > 0.70 else 0)
df['Renda_Per_Capita'] = df['MonthlyIncome'] / (df['NumberOfDependents']+1)
df['Total_Atrasos'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] +
                       df['NumberOfTime60-89DaysPastDueNotWorse'] +
                       df['NumberOfTimes90DaysLate'])
df['Uso_Cartao_Critico'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(lambda x: 1 if x>=1.0 else 0)
````
![alt text](<Uso cartão.png>)

## 4. Modelagem

### 4.1 Seleção de variáveis
````
features_base = [
    'RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse',
    'MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents','Renda_Per_Capita','Total_Atrasos','Uso_Cartao_Critico'
]
````
### 4.2 Treinamento
````
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_A = df[features_base+['DebtRatio_Clean']]
y_A = df['Target_Inadimplencia']

X_B = df[features_base]
y_B = df['Target_Superendividado']

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A,y_A,test_size=0.3,random_state=42)
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B,y_B,test_size=0.3,random_state=42)

clf_A = RandomForestClassifier(n_estimators=50,max_depth=10,class_weight='balanced',random_state=42)
clf_A.fit(X_train_A,y_train_A)

clf_B = RandomForestClassifier(n_estimators=50,max_depth=10,class_weight='balanced',random_state=42)
clf_B.fit(X_train_B,y_train_B)
````

## 5. Avaliação

### 5.1 Métricas
````
from sklearn.metrics import roc_auc_score
nota_A = roc_auc_score(y_test_A, clf_A.predict_proba(X_test_A)[:,1])
nota_B = roc_auc_score(y_test_B, clf_B.predict_proba(X_test_B)[:,1])
print(nota_A, nota_B)
````

### 5.2 Matrizes de confusão
````
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true,y_pred,title):
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test_A, clf_A.predict(X_test_A), "Matriz Inadimplência")
plot_confusion_matrix(y_test_B, clf_B.predict(X_test_B), "Matriz Superendividamento")
````

![alt text](<Matriz de confusão.png>)
![alt text](<Matriz de confusão superendividamento.png>)



## 6. Deployment (Implantação)
- API: FastAPI com endpoint /analisar_credito.
- Interface: Streamlit para entrada de dados e visualização dos resultados.
- Artefatos: modelos .pkl e regras salvos com joblib.
````
import joblib
joblib.dump(clf_A,'modelo_inadimplencia.pkl')
joblib.dump(clf_B,'modelo_superendividamento.pkl')
````
![alt text](image.png)
![alt text](image-1.png)