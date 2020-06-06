""" TODO:
    - Ojo con categorías no observadas: cambiar a drop_first a 'False'
    - Método pipeline?
    - Mètodo Actualizar
    - Atributo mejores parámetros

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from time import time


class PrepML:

    def __init__(self, df):
        self.df = df
        self.columns = list(df.columns)
        self.prep_objects = []

    def one_hot_encoder(self, columns, drop_first=True):
        """
        Recodifica las columnas seleccionadas (variables categóricas), creando k o k-1 nuevas columnas
            por cada clase que posea la columna original, imputando con valores 1 y 0 según si en el registro
            se presenta o no la categoría.

        :param columns: [list] lista de columnas del df que se desean procesar por el encoder
        :param drop_first: [bool]
        :return: df preprocesado
        """

        aux = {'drop': {True: 'first', False: None},
               'unknown': {True: 'error', False: 'ignore'}
               }
        # Categorías para one-hot
        df_oh = self.df[columns]
        categories = [list(df_oh[var]
                           .value_counts()
                           .sort_values(ascending=False)
                           .index)
                      for var in df_oh]
        # Nombre de columnas dummy
        tuples = [(var, list(df_oh[var]
                             .value_counts()
                             .sort_values(ascending=False)
                             .index)[1:]
                   ) for var in df_oh]
        dummy_names = ['{}_{}'.format(tup[0], cat) for tup in tuples for cat in tup[1]]
        dummy_names = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dummy_names]

        # Instanciamos  objeto de preproceso
        oh_enc = OneHotEncoder(categories,
                               sparse=True,
                               drop=aux['drop'][drop_first],
                               handle_unknown=aux['unknown'][drop_first])
        # Entrenamos y transformamos columnas con el encoder
        dummy_data = oh_enc.fit_transform(df_oh)
        prep_df = pd.DataFrame.sparse.from_spmatrix(data=dummy_data,
                                                    columns=dummy_names)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.prep_objects += [{'onehot': oh_enc}]

        return self.df

    def standard_scaler(self, columns):
        """
        Recodifica las columnas seleccionadas (variables continuas) escalando sus valores a través
            de la transformación: (x - mean(X)) / std(X)
        :param columns:
        :return:
        """

        # Instanciamos y entrenamos/transformamos con objeto de preproceso
        std_enc = StandardScaler()
        std_data = std_enc.fit_transform(self.df[columns])
        prep_df = pd.DataFrame(data=std_data,
                               columns=columns)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.prep_objects += [{'std_scaler': std_enc}]

        return self.df

    def transform_columns(self, transformer_instance, transformer_name, columns):

        # Instanciamos y entrenamos/transformamos con objeto de preproceso
        data = transformer_instance.fit_transform(self.df[columns])
        prep_df = pd.DataFrame(data=data,
                               columns=columns)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.prep_objects += [{transformer_name: transformer_instance}]

        return self.df

    def remove_outliers(self, columns, multiplier=1.5):

        Q1 = self.df[columns].quantile(0.25)
        Q3 = self.df[columns].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - multiplier * IQR)) |
                            (self.df > (Q3 + multiplier * IQR))
                            ).any(axis=1)].reset_index(drop=True).copy()
        return self.df

    def log_transformer(self, column):

        self.df[column] = self.df[column].map(lambda x: np.log(x))

       #list[{'encoder': params}]

    def pipeline(self, enc_dict):
        dc = {'onehot': self.one_hot_encoder,
              'std_scaler': self.standard_scaler,
              'rm_outliers': self.remove_outliers}
    #funcioniones(**dc)

    def to_train_test_samples(self, sample_col, target):

        start = time()

        df_train = self.df[self.df[sample_col] == 'train']
        df_test = self.df[self.df[sample_col] == 'test']

        X_train = df_train.drop(columns=[target, sample_col])
        y_train = df_train[target]

        X_test = df_test.drop(columns=[target, sample_col])
        y_test = df_test[target]

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        return X_train, y_train, X_test, y_test


class MLModel:

    def __init__(self, model):
        self.model = model
        self.best_model = None
        self.target = None
        self.features = None

    def fit(self, x_train, y_train):

        start = time()
        self.best_model = self.model.fit(x_train, y_train)
        self.target = y_train.name
        self.features = x_train.columns

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

    def grid_search(self, x_train, y_train, param_grid, cv=5):

        start = time()
        grid = GridSearchCV(estimator=self.model,
                            param_grid=param_grid,
                            n_jobs=-1,
                            cv=cv)
        if isinstance(self.model, XGBRegressor):
            grid.fit(x_train.values, y_train)
        else:
            grid.fit(x_train, y_train)

        print(f'Mejores parámetros:\n{grid.best_params_}\n')
        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        # Actualzación de atributos
        self.best_model = grid.best_estimator_
        self.target = y_train.name
        self.features = x_train.columns

    def metrics(self, x_test, y_test, print_results=True):

        if isinstance(self.model, XGBRegressor):
            y_hat = self.best_model.predict(x_test.values)
        else:
            y_hat = self.best_model.predict(x_test)

        metrics = {'mse': round(mean_squared_error(y_true=y_test,
                                                   y_pred=y_hat), 3),
                   'mae': round(mean_absolute_error(y_true=y_test,
                                                    y_pred=y_hat), 3),
                   'r2': round(r2_score(y_true=y_test,
                                        y_pred=y_hat), 3)}
        if print_results:
            for key, value in metrics.items():
                print('{}: {}'.format(key, value))

        return metrics

    def feature_importances(self, x_train):

        columns = list(self.features)
        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(data=self.best_model.feature_importances_.round(3),
                             index=columns).sort_values(ascending=False)
        else:
            raise ValueError("El algoritmo no tiene el atributo feature_importances")



