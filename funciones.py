import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rcParams
from ml_classes import MLModel


#Definición función para graficar atributos continuos.
def distrbution_graph(a):
    
    rcParams['figure.figsize'] = 8, 6
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(a, ax=ax_box)
    sns.distplot(a, ax=ax_hist,kde=False,bins=200,color="tomato")
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)


#definicion función para gráficos: count del atributo y boxplot con relacion con vector objetivo.
def count_box_plot(a, df, b=100000, c=True):
    
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
    
    cols = ['AirBagLocFront', 'BodyClass', 'BusFloorConfigType', 'BusType',
       'CustomMotorcycleType', 'DisplacementCC', 'DisplacementCI',
       'DisplacementL', 'Doors', 'EngineCylinders', 'EngineHP', 'EngineKW',
       'ErrorCode', 'ErrorText', 'FuelTypePrimary', 'Make', 'Manufacturer',
       'ManufacturerId', 'Model', 'ModelYear', 'MotorcycleChassisType',
       'MotorcycleSuspensionType', 'PlantCity', 'PlantCountry', 'TPMS',
       'TrailerBodyType', 'TrailerType', 'VIN', 'VehicleType']
    
    return [value for col, value in dict.items() if col in cols]


def train_mlmodels(model_list, grid_list, samples, category):

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

    # Enlistamos los archivos de modelos serialisados
    pickle_files = [f'{category}_{model.__class__.__name__.lower()}.sav' for model in model_list]

    for pickle_model in pickle_files:
        # Importamos mejor modelo
        best_model = MLModel.from_pickle(f'best_models/{pickle_model}')
        # Métricas mejor modelo
        print(pickle_model[:-4])
        print(best_model.train_test_metrics(samples[category]['X_train'],
                                            samples[category]['y_train'],
                                            samples[category]['X_test'],
                                            samples[category]['y_test']))
        print('\n')
