""" TODO:
    - Ojo con categorías no observadas: cambiar a drop_first a 'False'
    - Atributo mejores parámetros
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from time import time
import pickle
import random


class PrepML:
    """
    Clase para realizar el preproces de los datos requerido para los modelos de Machine Learning
    """
    def __init__(self, df):
        self.df = df.dropna()
        self.columns = list(df.columns)
        self.transformers = []
        self.df_ct = self.clean_categories()

    def clean_categories(self):
        """
        Convierte los caracterés no alfanúmericos en guiones de todas las columnas del DataFrame de tipo 'object'
        :return: DataFrame con columnas 'object' con valores convertidos
        """
        
        df_ct = self.df
        for var in df_ct.select_dtypes('object').columns:
            df_ct[var] = df_ct[var].map(lambda x: "".join(c if c.isalnum() else "_" for c in str(x)))

        return df_ct

    def one_hot_encoder(self, columns, drop_first=True):
        """
        Recodifica las columnas seleccionadas (variables categóricas), creando k o k-1 nuevas columnas
            por cada clase que posea la columna original, imputando con valores 1 y 0 según si en el registro
            se presenta o no la categoría.

        :param columns: [list] lista de columnas del df que se desean procesar por el encoder
        :param drop_first: [bool]
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
        self.transformers += [('onehot', oh_enc, columns)]

    def standard_scaler(self, columns):
        """
        Recodifica las columnas seleccionadas (variables continuas) escalando sus valores a través
            de la transformación: (x - mean(X)) / std(X)
        :param columns: columnas seleccionadas
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
        self.transformers += [('std_scaler', std_enc, columns)]

    def transform_columns(self, transformer_instance, transformer_name, columns):
        """
        Realiza en el DataFrame la instancia de la transformación asignada
        :param transformer_instance: intancia de la clase que genera la transformación
        :param transformer_name: nombre a asignar al transformador
        :param columns: columnas a aplicar la transformación
        """

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
        self.transformers += [(transformer_name, transformer_instance, columns)]

    def remove_outliers(self, columns, sample_col, iqr_multiplier=1.5, print_diff=False):
        """
        Remove los outliers de las columnas númericas según el critero de los boxplot de John Tucky
        :param columns: columnas para buscar outliers
        :param sample_col: columna que asigna el tipo de muestra
        :param iqr_multiplier: multiplicador del rango intercuartil
        :param print_diff: imprimir diferencias entre la muestra de entrenamiento de antes y después
                            de la eliminación de outliers.
        """

        train = self.df[self.df[sample_col] == 'train'].reset_index(drop=True)
        test = self.df[self.df[sample_col] == 'test'].reset_index(drop=True)

        Q1 = train[columns].quantile(0.25)
        Q3 = train[columns].quantile(0.75)
        IQR = Q3 - Q1
        train_af = train[~((train < (Q1 - iqr_multiplier * IQR)) |
                            (train > (Q3 + iqr_multiplier * IQR))
                            ).any(axis=1)].reset_index(drop=True).copy()

        self.df = pd.concat([train_af, test], axis=0).reset_index(drop=True)
        self.df_ct = pd.concat([train_af, test], axis=0).reset_index(drop=True)

        if print_diff:
            #  Cálculo de diferencia en el tamaño de la muestra de entrenamiento
            before = train.shape[0]
            after = train_af.shape[0]
            print(f'Datos para entrenamiento antes de eliminación de outliers: {before}')
            print(f'Datos para entrenamiento después eliminación de outliers: {after}')
            print(f'Proporción de datos para entrenamiento eliminada: {1 - round(after / before, 3)}')

    def log_transformer(self, column):
        """
        Realiza una transformación logística a las columnas seleccionadas
        :param column:  columnas seleccionadas a transformar
        """

        self.df[column] = self.df[column].map(lambda x: np.log(x))

    def to_ml_samples(self, sample_col, target, test_size=.3, random_state=42):
        """
        Divide la base en 6 muestras: de entrenamiento, de prueba y de validación.
        :param sample_col: variable que clasifica el tipo de muestra
        :param target: variable objetivo
        :param test_size: dimensiones de la muestra de prueba
        :param random_state: semilla pseudo-aleatoria
        :return: X_train, y_train, X_test, y_test, X_val, y_val
        """

        start = time()
        random.seed(random_state)

        df_val = self.df[self.df[sample_col] == 'test'].reset_index(drop=True)

        ix = self.df[self.df['sample'] == 'train'].index
        test_number = int(np.floor(len(ix) * test_size))
        test_ix = random.choices(ix, k=test_number)
        train_ix = list(set(ix) - set(test_ix))
        df_test = self.df.loc[test_ix].reset_index(drop=True)
        df_train = self.df.loc[train_ix].reset_index(drop=True)

        X_train = df_train.drop(columns=[target, sample_col])
        y_train = df_train.pop(target)
        X_test = df_test.drop(columns=[target, sample_col])
        y_test = df_test.pop(target)
        X_val = df_val.drop(columns=[target, sample_col])
        y_val = df_val.pop(target)

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        return [X_train, y_train, X_test, y_test, X_val, y_val]


class MLModel:
    """
    Clase que engloba las principales tareas del flujo de trabajo de Machine Learning para
        entrenar, evaluar y exportar modelos.
    """

    def __init__(self, model):
        self.model = model
        self.best_model = None
        self.target = None
        self.features = None

    @classmethod
    def from_pickle(cls, filepath):
        """
           Constructor alternativo de la clase, para instanciarla a partir de un modelo en pickle
        """
        best_model = pickle.load(open(filepath, 'rb'))

        obj = cls.__new__(cls)
        super(MLModel, obj).__init__()
        obj.model = None
        obj.best_model = best_model

        return obj

    def fit(self, x_train, y_train):
        """
        Entrena el modelo
        :param x_train: Atributos de entrenamiento
        :param y_train: Variable Objetivo de entrenamiento
        """

        start = time()
        self.best_model = self.model.fit(x_train, y_train)
        self.target = y_train.name
        self.features = x_train.columns

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

    def grid_search(self, x_train, y_train, param_grid, cv=5, n_jobs=-2):
        """
        Realiza la optimización de hiperparámetros a partir de la grilla definida
        :param x_train: Atributos de entrenamiento
        :param y_train: Variable objetivo de entrenamiento
        :param param_grid: Grilla de hiperparámetros
        :param cv: Número de validaciones cruzadas a realizar
        :param n_jobs: número de trabajos a realizar en paralelo
        """
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
        """
        Devuelve las principales métricas utilizadas para un modelo de regeresión
        :param x_test: Atributos de muestra de prueba
        :param y_test: Variable objetivo de muestra de prueba
        :param print_results: si desea que los resultados sean impresos
        :return: diccionario con las métricas (RSME, MAE y R2)
        """

        if isinstance(self.best_model, XGBRegressor):
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
        """
        Devuelve las principales métricas utilizadas para evaluar modelos de regeresión
            para las muestra de entrenamiento y de prueba.
        :param X_train: Atributos de la muestra de entrenamiento
        :param y_train: Variable objetivo de la muestra de entrenamiento
        :param X_test: Atributos de la muestra de prueba
        :param y_test: Variable objetivo de la muestra de prueba
        :return: DataFrame con las métricas
        """

        train_met = self.metrics(X_train, y_train, print_results=False)
        test_met = self.metrics(X_test, y_test, print_results=False)
        data = [[val for key, val in train_met.items()],
                [val for key, val in test_met.items()]
                ]
        cols = ['RSME', 'MAE', 'R2']
        ix = ['Train', 'Test']

        return pd.DataFrame(data=data, columns=cols, index=ix)

    def feature_importances(self, columns_name):
        """
        Crea un objeto Series con los atributos más relevantes
        :param columns_name: Nombre de las columnas de la muestra de entrenamiento
        :return: Series con Atributos más relevantes
        """

        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(data=self.best_model.feature_importances_,
                             index=columns_name).sort_values(ascending=False)
        else:
            raise ValueError("El algoritmo no tiene el atributo feature_importances")

    def to_pickle(self, car_category):
        """
        Serializa el mejor modelo y guarda el archivo en el sistema
        :param car_category: categoría del vehículo del modelo
        """

        model_name = self.best_model.__class__.__name__.lower()
        pickle.dump(self.best_model, open(f'best_models/{car_category}_{model_name}.sav', 'wb'))

    def to_pipeline(self, transformers, X_ct):
        """
        Crea un pipeline con el mejor modelo y la lista transformadores asignado
        :param transformers: lista con tuplas de los trasnformadores
        :param X_ct: Atributos para entrenar el objeto ColumnTransformer
        :return: pipeline instanciado
        """

        col_tf = ColumnTransformer(transformers).fit(X_ct)
        pipeline = Pipeline([
            ('preprocessor', col_tf),
            ('model', self.best_model)
        ])

        return pipeline




