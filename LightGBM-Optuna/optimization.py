########################################################################################################
# -= PACOTES UTILIZADOS =- #

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import optuna


########################################################################################################
# -= PARÂMETROS DE REPRODUCIBILIDADE DOS EXPERIMENTOS =- #

# Fixando parâmetros de reproducibilidade
seed = 1432

# Fixando uma semente aleatória diretamento no Numpy
np.random.seed(seed=seed)

# Definindo uma semente aleatória no amostrador de hiperparâmetros do Optuna
tpesamper = optuna.samplers.TPESampler(seed=seed)


########################################################################################################
# -= CARGA DO CONJUNTO DE DADOS =- #

def load_data():

  # Carregando os dados em um DataFrame do Pandas
  url   = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
  dados = pd.read_csv(url)

  # LabelEnconder nos dados de saída
  dados['variety'].replace({"Setosa": 0, "Versicolor": 1, "Virginica": 2}, inplace=True)

  # Separando os dados de entrada dos dados de saída
  X, Y = dados.drop(columns=["variety"]), dados["variety"]

  # Retornando os dados para modelagem
  return X, Y

# Chamando a função load_data()
X, Y = load_data()


########################################################################################################
# -= OTIMIZANDO HIPERPARÂMETROS DO ALGORITMO LGBM COM O OPTUNA =- #

def fit_lgbm(trial, train, valid):

  # Desempacotando os dados para treino e validação
  X_train, Y_train = train
  X_valid, Y_valid = valid

  # Transformando os dados em um tipo específico do LGBM
  dtrain = lgb.Dataset(data=X_train, label=Y_train)
  dvalid = lgb.Dataset(data=X_valid, label=Y_valid)

  # Definindo espaço de pesquisa dos hiperparâmetros
  params = {
      "objective":        "multiclass",
      "metric":           "multi_logloss",
      "boosting":         "gbdt",
      "verbosity":        -1,
      "num_class":        3,
      "seed":             seed,
      "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1),
      "num_leaves":       trial.suggest_int("num_leaves", 2, 256),
      "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
      "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
      "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
      "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
      "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10)
  }

  # Instanciando o pruning integrado
  pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss", valid_name="valid_1")

  # Ajustando o algoritmo de Aumento de Gradiente aos dados
  modelo = lgb.train(
      params=params,
      train_set=dtrain,
      valid_sets=[dtrain, dvalid],
      early_stopping_rounds=20,
      callbacks=[pruning_callback]
  )

  # Armazenando os resultado de treino e validação do modelo
  log = {
      "train/multi_logloss": modelo.best_score["training"]["multi_logloss"],
      "valid/multi_logloss": modelo.best_score["valid_1"]["multi_logloss"]
  }

  # Retornando os resultados salvos na variável "log"
  return log


def objective(trial):

  # Instanciando o KFold para validação cruzada
  kf = KFold(n_splits=5, shuffle=True, random_state=seed)

  # Criando uma variável responsável por armazenar o erro de ajuste do modelo aos dados de validação
  valid_score = 0

  # Aplicando uma validação cruzada
  for train_idx, valid_idx in kf.split(X, Y):

    # Empacotando os dados para treino e validação
    train_data = X.iloc[train_idx], Y[train_idx]
    valid_data = X.iloc[valid_idx], Y[valid_idx]

    # Chamando a função fit_lgbm()
    log = fit_lgbm(trial, train_data, valid_data)

    # Armazenando o valor de erro do modelo para os dados de validação
    valid_score += log["valid/multi_logloss"]

  # Retornando a média logloss
  return valid_score / 5

# Tempo de execução
tempo = 60 * 60 * 0.1 # 6 minutos

# Definindo objeto de estudo
study = optuna.create_study(sampler=tpesamper, pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, timeout=tempo)


########################################################################################################
# -= AVALIAÇÃO DOS RESULTADOS DA OTIMIZAÇÃO =- #

# Instanciando os melhores parâmetros encontrados
params         = study.best_params
params["seed"] = seed

# Imprimindo os resultados da otimização
print("-" * 35)
print(f"Best Score: {study.best_value:.2f}")
print("Best Params ⇓")
print("-" * 35)
params


########################################################################################################
# -= TREINANDO MODELO E APLICANDO UMA VALIDAÇÃO CRUZADA COM OS MELHORES HIPERPARÂMETROS ENCONTRADOS =- #

# Parâmetros necessários para o processo de treino e validação do modelo
valid_score = 1
previsoes   = np.zeros(X.shape[0])
kf          = KFold(n_splits=5, random_state=seed, shuffle=True)
n           = 1
acc         = []

# Aplicando uma validação cruzada com os melhores parâmetros encontrados
for train_idx, valid_idx in kf.split(X, Y):

  # Organizando os dados para treino e validação
  X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
  X_valid, Y_valid = X.iloc[valid_idx], Y.iloc[valid_idx]

  # Instanciando e treinando um estimador LightGBM
  modelo = LGBMClassifier(**params)
  modelo.fit(X_train, Y_train)

  # Gerando previsões com os dados de validação
  prev_valid = modelo.predict(X_valid)

  # Armazenando os valores previsto
  previsoes[valid_idx] = prev_valid

  # Acurácia i-fold
  acc.append(accuracy_score(Y_valid, prev_valid))

  # Relatório de classificação para cada um dos 5-Folds
  print("-=" * 27)
  print(f"Relatório - {n}\n{classification_report(Y_valid, prev_valid)}")
  print("-=" * 27)
  n += 1

# Resultados finais
print("-=" * 20)
print(f"Accuracy Score: {(accuracy_score(Y, previsoes) * 100):.2f}")
print(f"Desvio padrão: {np.std(acc):.2f}")
print("-=" * 20)