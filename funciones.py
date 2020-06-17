import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab
import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
from ml_classes import MLModel

def distrbution_graph(a):
    """
    Función para graficar atributos continuos.
    """
    
    rcParams['figure.figsize'] = 8, 6
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(a, ax=ax_box)
    sns.distplot(a, ax=ax_hist,kde=False,bins=200,color="tomato")
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)



def count_box_plot(a, df, b=100000, c=True):
    """
    Función para generar countplot del atributo y boxplot con relacion con vector objetivo.
    """
    
    rcParams['figure.figsize'] = 14, 6
    sns.set(style="darkgrid")
    plt.subplot(1,2,1)
    ax = sns.countplot(x=a, data=df)
    plt.xticks(rotation=45)
    plt.title(a,fontsize=15)
    plt.xlabel("")
    plt.ylabel("number of cars")
    plt.xticks(rotation=30, fontsize=8)
    plt.axis(c)
    plt.subplot(1,2,2)
    sns.boxplot(x=df[a], y=df['Price'])#, order=list(sorted_nb.index))
    plt.title(a,fontsize=15)
    plt.xticks(rotation=30,fontsize=8)
    plt.ylim(0,b)
    plt.axis(c)


def get_info(dict):
    """
    Función auxiliar para filtrar la API
    """
    
    cols = ['AirBagLocFront', 'BodyClass', 'BusFloorConfigType', 'BusType',
       'CustomMotorcycleType', 'DisplacementCC', 'DisplacementCI',
       'DisplacementL', 'Doors', 'EngineCylinders', 'EngineHP', 'EngineKW',
       'ErrorCode', 'ErrorText', 'FuelTypePrimary', 'Make', 'Manufacturer',
       'ManufacturerId', 'Model', 'ModelYear', 'MotorcycleChassisType',
       'MotorcycleSuspensionType', 'PlantCity', 'PlantCountry', 'TPMS',
       'TrailerBodyType', 'TrailerType', 'VIN', 'VehicleType']
    
    return [value for col, value in dict.items() if col in cols]


def train_mlmodels(model_list, grid_list, samples, category):
    """
    Función auxiliar para entrenar una serie de modelos según la catgoría de vehículo
    """

    for model, grid in zip(model_list, grid_list):
        print(f'{category}_{model.__class__.__name__.lower()}')
        # Instanciamos Clase auxiliar para entrenar, ajustar y evaluar modelos de ML
        model_reg = MLModel(model=model)
        # Implementación del grid search
        model_reg.grid_search(samples[category]['X_train'],
                              samples[category]['y_train'],
                              param_grid=grid,
                              n_jobs=-2,
                              cv=5)
        # Serialización del mejor modelo
        model_reg.to_pickle(car_category=category)
        print('\n')


def metrics_pickled_mlmodels(model_list, samples, category):
    
    """
    Función auxiliar para serializar modelos según la catgoría de vehículo
    """

    # Enlistamos los archivos de modelos serialisados
    pickle_files = [f'{category}_{model.__class__.__name__.lower()}.sav' for model in model_list]

    for pickle_model in pickle_files:
        # Importamos mejor modelo
        best_model = MLModel.from_pickle(f'best_models/{pickle_model}')
        # Métricas mejor modelo
        print(pickle_model[:-4])
        print(best_model.train_val_metrics(samples[category]['X_train'],
                                            samples[category]['y_train'],
                                            samples[category]['X_val'],
                                            samples[category]['y_val']))
        print('\n')


# función para crear qq plot
def qq_plot(model_list, X_val_list, y_val_list):
    """
    @parametros: model_list= Listado con mejores modelos.
                 X_val_list= Listados con los atributos para la validación para cada modelos.
                 y_val_list= Listado con el vector objetivo para la validación de cada modelo.

    @retorno: 1- grafico qq-plot que muestra la distribución de los errores comparados con la distribución normal.

    """
    preds_0 = pd.DataFrame({"preds": model_list[0].predict(X_val_list[0]), "true": y_val_list[0]})
    preds_0["residuals"] = preds_0["true"] - preds_0["preds"]
    preds_1 = pd.DataFrame({"preds": model_list[1].predict(X_val_list[1]), "true": y_val_list[1]})
    preds_1["residuals"] = preds_1["true"] - preds_1["preds"]
    preds_2 = pd.DataFrame({"preds": model_list[2].predict(X_val_list[2]), "true": y_val_list[2]})
    preds_2["residuals"] = preds_2["true"] - preds_2["preds"]
    preds_3 = pd.DataFrame({"preds": model_list[3].predict(X_val_list[3]), "true": y_val_list[3]})
    preds_3["residuals"] = preds_3["true"] - preds_3["preds"]
    rcParams['figure.figsize'] = 12, 3
    grid = plt.GridSpec(1, 4, wspace=0.6, hspace=0.3)
    plt.subplot(grid[0, 0])
    stats.probplot(abs(preds_0["residuals"]), dist="norm", plot=pylab)
    plt.title(f"Probability plot ALL types")
    plt.subplot(grid[0, 1])
    stats.probplot(abs(preds_1["residuals"]), dist="norm", plot=pylab)
    plt.title(f"Probability plot PSG")
    plt.subplot(grid[0, 2])
    stats.probplot(abs(preds_2["residuals"]), dist="norm", plot=pylab)
    plt.title(f"Probability plot MPP")
    plt.subplot(grid[0, 3]);
    stats.probplot(abs(preds_3["residuals"]), dist="norm", plot=pylab)
    plt.title(f"Probability plot TRK")


def grafico_importancia(fit_model, feat_names):
    """
    @parametros: fit_model= modelo al que se quiere conocer la importancia de los atributos.
                 feat_names=Listado con el nombre de los atributos del modelo.

    @retorno: 1- Gráfico con los 10 atributos más importanotes del modelo.
    """
    tmp_importance = fit_model.feature_importances_
    sort_importance = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importance[0:9]]
    plt.title("Feature importance")
    plt.barh(range(len(names)), tmp_importance[sort_importance[0:9]])
    plt.yticks(range(len(names)), names, rotation=0)
