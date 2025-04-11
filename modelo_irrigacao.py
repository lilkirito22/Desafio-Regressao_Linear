# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, kstest, probplot
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# %% [markdown]
# ### Carregar e visualizar dados 

# %%
# Carrregar dados de um CSV

df_fazenda = pd.read_csv('./datasets/dados_de_irrigacao.csv')

# %%
df_fazenda.info()

# %%
df_fazenda.head(5)

# %% [markdown]
# ### Analise exploratoria de dados EDA

# %%
df_fazenda.describe()

# %%
sns.scatterplot(data=df_fazenda, x='Horas de Irrigação', y='Área Irrigada por Ângulo')

# %%
sns.heatmap(df_fazenda.corr('pearson'), annot=True)

# %%
sns.heatmap(df_fazenda.corr('spearman'), annot=True)


# %% [markdown]
# ### Treinamento do Modelo

# %%
#dividir os dados em Variavel independente e dpeendente e depois separar os conjuntos de treino e teste
# so ressaltando para mim mesmo, o uso do reshape ou [[]] é para quando eu tiver passando apenas uma coluna, no caso da variavel dependente (y) ele nao é necessariamente necessario porem como vai fazer uso de predict entao é necessario
X = df_fazenda[['Horas de Irrigação']]
y = df_fazenda[['Área Irrigada por Ângulo']]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.4,random_state=50)




# %%
#Criando o modelo
reg_model = LinearRegression()

# %%
#treinando o modelo
reg_model.fit(X_train, y_train)

# %%
reg_model.coef_

# %%


# %%
reg_model.intercept_

# %%
sns.scatterplot(data=df_fazenda, x='Horas de Irrigação', y='Área Irrigada por Ângulo')
plt.plot(df_fazenda['Horas de Irrigação'], reg_model.predict(df_fazenda[['Horas de Irrigação']]), color='red')

# %% [markdown]
# ### Validar Modelo 

# %%
#Fazer uma predição com base no conjunto de testes

y_pred = reg_model.predict(X_test)

# %%
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

# %%
r2_score(y_test, y_pred)

# %%
# Calcular metrica MAE mean absolute error
# MAE = media (y_test - y_pred)
# É uma metrica facil de interpetar, MAE é menos sensivel a outliers
mean_absolute_error(y_test, y_pred)

# %%
# Calcular metrica MSE (mean squared error)
#MSE = media (y_test - y_pred)^2
#nao é uma metrica facil de interpretar
# é uma metrica mais sensivel a outliers e penaliza grandes erros
mean_squared_error(y_test, y_pred)

# %%
# evitar error do mean_squared_error com squared false
from sklearn.metrics import root_mean_squared_error

# %%
#Calcular metrica RMSE (square root mean squared error)
#RMSE = media  (raiz (y_test - y_pred)^2)
#É uma metrica mais facil de interpretar
# mas ainda sim continua mais sensivel aos outliers
root_mean_squared_error(y_test,y_pred)

# %%
x_axis = range(len(y_test))


# %%
plt.Figure(figsize=(10,6))
sns.scatterplot(x=x_axis, y=y_test.values.reshape(-1), color='blue', label='Reais')

sns.scatterplot(x=x_axis, y=y_pred.reshape(-1), color='red', label='Preditos')

plt.legend()
plt.show

# %%
import numpy as np

# %%
# Métricas de desempenho
mse = np.mean((y_test - y_pred)**2)
mae = np.mean(np.abs(y_test - y_pred))
print(f"MSE: {mse}")
print(f"MAE: {mae}")

# %%
#Fazendo a predição pedida de exemplo
reg_model.predict([[15]])

# %%



