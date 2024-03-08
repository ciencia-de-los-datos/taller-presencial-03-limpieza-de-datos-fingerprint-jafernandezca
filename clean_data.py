"""Taller evaluable presencial"""

import nltk
import pandas as pd
from nltk.stem import PorterStemmer


nltk.download('punkt')

def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    df = pd.read_csv(input_file)
     
    return df


def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

     # 1. Copie la columna 'text' a la columna 'fingerprint'
    df['fingerprint'] = df['text']

    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    df['fingerprint'] = df['fingerprint'].str.strip()

    # 3. Convierta el texto a minúsculas
    df['fingerprint'] = df['fingerprint'].str.lower()

    # 4. Transforme palabras que pueden (o no) contener guiones por su versión sin guion
    df['fingerprint'] = df['fingerprint'].str.replace(r'(\w+)-(\w+)', r'\1\2', regex=True)

    # 5. Remueva puntuación y caracteres de control
    df['fingerprint'] = df['fingerprint'].apply(lambda x: ''.join([char for char in x if char.isalnum() or char.isspace()]))

    # 6. Convierta el texto a una lista de tokens
    df['fingerprint'] = df['fingerprint'].apply(nltk.word_tokenize)

    # 7. Transforme cada palabra con un stemmer de Porter
    stemmer = PorterStemmer()
    df['fingerprint'] = df['fingerprint'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

    # 8. Ordene la lista de tokens y remueve duplicados
    df['fingerprint'] = df['fingerprint'].apply(lambda tokens: sorted(set(tokens)))

    # 9. Convierta la lista de tokens a una cadena de texto separada por espacios
    df['fingerprint'] = df['fingerprint'].apply(lambda tokens: ' '.join(tokens))


    return df





def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()

    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    df = df.sort_values(by=['fingerprint', 'text'])

    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    df_cleaned = df.groupby('fingerprint').first().reset_index()

    # 3. Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    fingerprint_dict = df_cleaned.set_index('fingerprint')['text'].to_dict()

    # 4. Cree la columna 'cleaned' usando el diccionario
    df['cleaned'] = df['fingerprint'].map(fingerprint_dict)

    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df['text'] = df['cleaned']
    df[['text']].to_csv(output_file, index=False)
    

def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )

