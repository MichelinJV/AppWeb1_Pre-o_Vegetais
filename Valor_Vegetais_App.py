# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:53:41 2021

@author: MichelinJV """


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import streamlit as st
import time

def main():
    df = load_data()
    df = conversao(df)
    
    ''' Função que cria o aplicativo'''
    
    # Insere um título
    st.title("Modelo Regressão: Preço de Vegetais")
   
    # texto de apresentação
    html_temp = """ Apliativo que avalia o preço do Quilo (Kg) dos vegetais: 1-batata, 
    2-tomate, 3-ervilha, 4-abóbora, 5-pepino, 6-abobrinha, 7-rabanete, 
    8-cabaça, 9-cebola, 10-alho, 11-repolho, 12-couve-flor, 13-pimenta, 
    14-quiabo, 15-berinjela, 16-gengibre
    """
    
    # insere o texto de apresentação
    st.markdown(html_temp)

    # Cria espaços para serem preenchidos pelo usuário com uma breve informação
    # sobre qual valor inserir
    
    st.warning("Somente Insira Valores Numéricos nos Campos Seguintes")
    
    var_1 = st.text_input("Escolha o número do vegetal listado acima")
    var_2 = st.text_input(
        "Escolha o número da estação (1-primavera,2-verão,3-outono,4-inverno)")
    var_3 = st.text_input("Escolha o número do mês (1-12)")
    var_4 = st.text_input("Digite o valor da temperatura")
    var_5 = st.text_input(
        "Aconteceu desastre nos últimos 3 meses (1-sim, 0-não)")
    var_6 = st.text_input(
        "Escolha o número da condição do vegetal (1-fresco,2-pedaço,3-avariado)")
    
    submit = st.button("Preço Previsto")
    if submit:
        if var_1 and var_2 and var_3 and var_4 and var_5 and var_6:
            with st.spinner("Calculando..."):
                time.sleep(2)
                var_1, var_2, var_3, var_4, var_5, var_6 = \
                    int(var_1), int(var_2), int(var_3), \
                    int(var_4), int(var_5), int(var_6)
                test = np.array([var_1, var_2, var_3, var_4, var_5, var_6])
                test = test.reshape(1,-1)
                prediction = predict(df, test)
                st.info(f"Preço Previsto do Quilo do Vegetal é {prediction}")
        else:
            st.error("Por Favor, Preencha Todos os Campos")
                                  
                
    
@st.cache
def load_data():
    return pd.read_csv('Vegetable_market_train.csv')

def train_model(df):
    global scaler 
    y = df['Preço_por_Kg']
    X = df.drop('Preço_por_Kg', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    model = XGBRegressor().fit(X,y)
    return model, scaler

def predict(df, test):
    model, scaler = train_model(df)
    X_test = scaler.transform(test)
    y_pred = model.predict(X_test)
    return np.round(y_pred, 2)
   
def conversao(df):
    ''' Função que converte os dados categóricos em ordinais '''

    df = df.copy()

    df.columns = ['Vegetais', 'Estação', 'Mês', 'Temperatura',
                  'Desastre_últimos_3_meses', 'Condição_vegetais',
                  'Preço_por_Kg']

    df['Vegetais'] = df['Vegetais'].replace({
        'potato': 1, 'tomato ': 2, 'peas': 3, 'pumkin': 4,
        'cucumber': 5, 'pointed grourd ': 6, 'Raddish': 7,
        'Bitter gourd': 8, 'onion': 9, 'garlic': 10, 'cabage': 11,
        'califlower': 12, 'chilly': 13, 'okra': 14, 'brinjal': 15,
        'ginger': 16, 'radish': 7
    })

    df['Estação'] = df['Estação'].replace({
        'winter': 1, 'summer': 2, 'monsoon': 0,
        'autumn': 3, 'spring': 4
    })

    df['Mês'] = df['Mês'].replace({
        'jan': 1,
        'apr': 4,
        'july': 7,
        'sept': 9,
        'oct': 10,
        'dec': 12,
        'may': 5,
        'aug': 8,
        'june': 6,
        ' ': np.NaN,
        'march': 3
    })

    df['Mês'] = df['Mês'].fillna(df['Mês'].mode()[0])

    df['Desastre_últimos_3_meses'] = df['Desastre_últimos_3_meses']\
    .replace({'no': 0,'yes': 1})

    df['Condição_vegetais'] = df['Condição_vegetais'].replace({
        'fresh': 1, 'scrap': 2, 'avarage': 3, 'scarp': 2
    })

    return df

if __name__ == '__main__':
    main() 