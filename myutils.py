### myutils.py ###
import numpy as np
import pandas as pd

#RANGO INTERCUARTILICO
def iqr_calculate(column):
    description = column.describe()
    iqr = description['75%']-description['25%']
    return iqr

#CONVIERTE DATAFRAMES A MATRICES NUMPY
def to_matrix(X):
    X_matrix=X
    
    if type(X)== pd.core.frame.DataFrame or type(X)== pd.core.series.Series:
        X_matrix=X_matrix.as_matrix()
        
    return X_matrix

#LIMITES VALORES ATIPICOS EXTREMOS O NO
def outliers_limits(column, extreme=True):
    col_matrix=to_matrix(column)
    minq,maxq=np.percentile(col_matrix,[25,75])
    iqr = maxq-minq
    coef=(extreme+1)*1.5
    outliers_limits = minq-coef*iqr,maxq+coef*iqr
    return outliers_limits

#ELIMINA VALORES ATIPICOS
def delete_outliers(X, y, extreme=True):
    limits = outliers_limits(y,extreme)
    
    X_no_outliers = X[y<=limits[1]]
    y_no_outliers = y[y<=limits[1]]
    
    X_no_outliers = X_no_outliers[y_no_outliers>=limits[0]]
    y_no_outliers = y_no_outliers[y_no_outliers>=limits[0]]
    
    return X_no_outliers,y_no_outliers

import seaborn as sns
"""
|----------------------------|
|FUNCIONES PARA VISUALIZACION|
|----------------------------|
"""

#MOSTRAR HISTOGRAMA PARA UNA COLUMNA DADA
def mostrar_histograma(column, bins=50, titulo='', etiqueta=None):
    print('Histograma', titulo, ':')
    fig, ax = plt.subplots()
    ax.hist(column, bins)
    ax.set_xlabel(etiqueta)
    plt.show()
    
#MUESTRA DIAGRAMA DE CAJAS
def mostrar_diagrama_cajas(column, titulo=''):
    print('Diagrama de cajas', titulo, ':')
    plt.boxplot(column, vert=False)
    plt.show()

#MUESTRA LA MATRIZ DE CORRELACIONES
def mostrar_matriz_correlacion(data):
    print('Matriz de correlaciones:')
    correlation=data.corr() #obtain the correlation matrix
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation,ax=ax)
    plt.show()

#MUESTRA MATRIZ CON VARIABLES DE FUERTE CORRELACION
def mostrar_matriz_correlacion_fuerte(data, valor = 0.7):  
    print('Matriz de correlaciones de variables con correlacion superior a ', valor, ':')
    correlation=data.corr() #obtain the correlation matrix
    aux=(abs(correlation)-np.identity(correlation.shape[0])).max() #maximum correlation of each variable
    selected_feats=aux[aux>valor].index#take only variables whose maximum correlation is strong.
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation.loc[selected_feats,selected_feats],ax=ax,annot=True,fmt='.2f')
    plt.show()
    
#MUESTRA MATRIZ DE CORRELACION CON LAS N VARIABLES MAS FUERTES
def mostrar_matriz_correlacion_variables_fuertes(data, variables = 20):  
    print('Matriz de correlaciones de las ', variables, ' variables mas fuertes:')
    correlation=data.corr()
    aux=abs(correlation['Poder_Adquisitivo']).sort_values(ascending=False) #sort variables by their correlation with SalePrice
    selected_feats=aux[1:variables+1].index
    sns.set()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(correlation.loc[selected_feats,selected_feats], annot=True,fmt='.2f',ax=ax)
    plt.show()

    
#MUESTRA LA MATRIZ DE DISPERSION
def mostrar_matriz_dispersion(column_a, column_b, label_a = None, label_b = None):
    fig, ax = plt.subplots()
    ax.scatter(column_a,column_b)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    plt.show()

#MUESTRA MATRIZ DE DISPERSION CON VARIABLES DE FUERTE CORRELACION
def mostrar_matriz_dispersion_fuerte(data, valor = 0.7):
    print('Matrices de dispersion de variables con correlacion superior a ', valor, ':')
    correlation = data.corr()
    aux=abs(correlation)-np.identity(correlation.shape[0]).max()
    selected_feats=aux[aux>valor].index
    sns.set(style="ticks")
    sns.pairplot(data[selected_feats])
    plt.show()
    
#MUESTRA MATRIZ DE DISPERSION DE LAS N VARIABLES MAS FUERTES
def mostrar_matriz_dispersion_variables_fuertes(data, variables = 20):
    print('Matrices de disperion de las ', variables, ' variables mas fuertes:')
    correlation = data.corr()
    aux=abs(correlation['Poder_Adquisitivo']).sort_values(ascending=False)
    selected_feats=aux[1:variables+1].index #Seleccionamos las variables mas fuertes excluyendo PA
    sns.set(style="ticks")
    sns.pairplot(data[selected_feats])
    plt.show()

#SELECCIONA BLOQUE DE DATOS
def mostrar_histograma_bloque(data,bloque = 'Imp_Cons'):
    for col in data.columns:
        if(col.find(bloque) != -1):
           #print(col)
           mostrar_histograma(data[col], 50, titulo=col, etiqueta=col)

#TRANSFORMACION LOGARITMICA
def transformacion_logaritmica(column):
    logarithmic_col = [np.log10(column)]
    return logarithmic_col

#CREA UNA VARIABLE RELACION
def crear_variable_relacion(column_a, column_b):
    new_column = [a / b for a,b in zip(column_a,column_b)]
    return new_column

############################################
def per_missing(X):
    missing_rows = X.isnull().sum()
    missing_rows_per = X.isnull().sum().apply(lambda x: 100*(x/len(X)) )
    return pd.concat([missing_rows,missing_rows_per],axis=1).sort_values(1,ascending=False)

def var_unicos(x):
    df1 = pd.DataFrame( x.apply(lambda x: x.nunique(), axis=0), columns=['Unicos'] )

    df1 = df1.reset_index()
    df2 = pd.DataFrame( x.dtypes , columns=['Tipo'] )
    df2 = df2.reset_index()
    return ( pd.concat([df1, df2['Tipo']],axis=1) ).sort_values(by=['Unicos'])
#sort_values() sort_index()