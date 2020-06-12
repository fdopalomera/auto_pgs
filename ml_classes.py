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
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from time import time
import pickle


class PrepML:

    def __init__(self, df):
        self.df = df.dropna()
        self.columns = list(df.columns)
        self.prep_objects = {}

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
        for var in columns:
            df_oh[var] = df_oh[var].map(lambda x: "".join(c if c.isalnum() else "_" for c in str(x)))
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
        self.prep_objects.update({'onehot': oh_enc})

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
        self.prep_objects.update({'std_scaler': std_enc})

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
        self.prep_objects.update({transformer_name: transformer_instance})

    def remove_outliers(self, columns, iqr_multiplier=1.5):

        Q1 = self.df[columns].quantile(0.25)
        Q3 = self.df[columns].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - iqr_multiplier * IQR)) |
                            (self.df > (Q3 + iqr_multiplier * IQR))
                            ).any(axis=1)].reset_index(drop=True).copy()

    def log_transformer(self, column):

        self.df[column] = self.df[column].map(lambda x: np.log(x))

    def express_pipeline(self, onehot_cols, stdscaler_cols, drop_first=True):

        self.one_hot_encoder(columns=onehot_cols,
                             drop_first=drop_first)
        self.standard_scaler(columns=stdscaler_cols)

    def to_train_test_samples(self, sample_col, target, test_size=.3, random_state=42):

        start = time()

        df_train = self.df[self.df[sample_col] == 'train']
        df_test = self.df[self.df[sample_col] == 'test']

        X_train, X_test, y_train, y_test = train_test_split(
                                            df_train.drop(columns=[target, sample_col]),
                                            df_train[target],
                                            test_size=test_size,
                                            random_state=random_state)
        X_val = df_test.drop(columns=[target, sample_col])
        y_val = df_test[target]

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        return X_train, y_train, X_test, y_test, X_val, y_val


class MLModel:

    def __init__(self, model):
        self.model = model
        self.best_model = None
        self.target = None
        self.features = None

    @classmethod
    def from_pickle(cls, filepath):
        best_model = pickle.load(open(filepath, 'rb'))

        obj = cls.__new__(cls)
        super(PrepML, obj).__init__()
        obj.model = None
        obj.best_model = best_model

        return obj

    def fit(self, x_train, y_train):

        start = time()
        self.best_model = self.model.fit(x_train, y_train)
        self.target = y_train.name
        self.features = x_train.columns

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

    def grid_search(self, x_train, y_train, param_grid, cv=5, n_jobs=1):
        start = time()
        grid = GridSearchCV(estimator=self.model,
                            param_grid=param_grid,
                            n_jobs=n_jobs,
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

    def metrics(self, x_test, y_test, print_results=False):

        if isinstance(self.model, XGBRegressor):
            y_hat = self.best_model.predict(x_test.values)
        else:
            y_hat = self.best_model.predict(x_test)

        metrics = {'rsme': round(np.sqrt(mean_squared_error(y_true=y_test,
                                                            y_pred=y_hat)), 1),
                   'mae': round(mean_absolute_error(y_true=y_test,
                                                    y_pred=y_hat), 1),
                   'r2': round(r2_score(y_true=y_test,
                                        y_pred=y_hat), 3)}
        if print_results:
            for key, value in metrics.items():
                print('{}: {}'.format(key, value))

        return metrics

    def train_test_metrics(self, X_train, y_train, X_test, y_test):

        train_met = self.metrics(X_train, y_train, print_results=False)
        test_met = self.metrics(X_test, y_test, print_results=False)
        data = [[val for key, val in train_met.items()],
                [val for key, val in test_met.items()]
                ]
        cols = ['RSME', 'MAE', 'R2']
        ix = ['Train', 'Test']
        return pd.DataFrame(data=data, columns=cols, index=ix)

    def feature_importances(self, x_train):

        columns = list(self.features)
        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(data=self.best_model.feature_importances_.round(3),
                             index=columns).sort_values(ascending=False)
        else:
            raise ValueError("El algoritmo no tiene el atributo feature_importances")



