import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class FlightPreprocessor:
    def __init__(self):
        self.label_encoders = {}
    
    def load_data(self, filepath='data/raw/flights_data.csv'):
        """Cargar datos crudos"""
        df = pd.read_csv(filepath)
        print(f"✓ Datos cargados: {len(df)} registros")
        return df
    
    def feature_engineering(self, df):
        """Crear nuevas características"""
        df = df.copy()
        
        # Convertir fechas
        df['fecha_salida'] = pd.to_datetime(df['fecha_salida'])
        
        # Características temporales
        df['dia_semana'] = df['fecha_salida'].dt.dayofweek
        df['mes'] = df['fecha_salida'].dt.month
        df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
        
        # Hora del día
        df['hora_salida_num'] = pd.to_datetime(df['hora_salida'], format='%H:%M').dt.hour
        df['periodo_dia'] = pd.cut(df['hora_salida_num'], 
                                    bins=[0, 6, 12, 18, 24], 
                                    labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
        
        # Métricas derivadas
        df['precio_por_hora'] = df['precio'] / (df['duracion_minutos'] / 60)
        df['score_escalas'] = 3 - df['escalas']
        
        print("✓ Feature engineering completado")
        return df
    
    def encode_categorical(self, df, columns):
        """Codificar variables categóricas"""
        df = df.copy()
        
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        print(f"✓ Variables categóricas codificadas: {columns}")
        return df
    
    def create_target_variable(self, df):
        """Crear variable objetivo: score de calidad del vuelo"""
        df = df.copy()
        
        # Normalizar precio (invertir: menor precio = mejor)
        precio_norm = 1 - (df['precio'] - df['precio'].min()) / (df['precio'].max() - df['precio'].min())
        
        # Normalizar duración (invertir: menor duración = mejor)
        duracion_norm = 1 - (df['duracion_minutos'] - df['duracion_minutos'].min()) / (df['duracion_minutos'].max() - df['duracion_minutos'].min())
        
        # Normalizar escalas (invertir: menos escalas = mejor)
        escalas_norm = 1 - (df['escalas'] / df['escalas'].max()) if df['escalas'].max() > 0 else 1
        
        # Score compuesto (pesos ajustables)
        df['score_vuelo'] = (
            0.5 * precio_norm +
            0.3 * duracion_norm +
            0.2 * escalas_norm
        )
        
        # Clasificación en categorías
        df['recomendacion'] = pd.cut(df['score_vuelo'], 
                                      bins=[0, 0.33, 0.66, 1.0],
                                      labels=['Regular', 'Bueno', 'Excelente'])
        
        print("✓ Variable objetivo creada")
        return df
    
    def prepare_for_ml(self, df):
        """Preparar dataset para ML"""
        categorical_features = ['origen', 'destino', 'aerolinea', 'clase', 'periodo_dia']
        
        # Codificar categóricas
        df = self.encode_categorical(df, categorical_features)
        
        # Seleccionar features para el modelo
        feature_columns = [
            'escalas', 'duracion_minutos', 'dias_anticipacion',
            'dia_semana', 'mes', 'es_fin_semana', 'hora_salida_num',
            'equipaje_incluido', 'asientos_disponibles',
            'origen_encoded', 'destino_encoded', 'aerolinea_encoded',
            'clase_encoded', 'periodo_dia_encoded'
        ]
        
        X = df[feature_columns]
        y = df['precio']
        
        print(f"✓ Dataset preparado: {X.shape[0]} muestras, {X.shape[1]} features")
        return X, y, df

if __name__ == "__main__":
    print("="*60)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    preprocessor = FlightPreprocessor()
    
    # Cargar datos
    df = preprocessor.load_data()
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Crear target
    df = preprocessor.create_target_variable(df)
    
    # Preparar para ML
    X, y, df_processed = preprocessor.prepare_for_ml(df)
    
    print(f"\n✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"\nDistribución de recomendaciones:")
    print(df_processed['recomendacion'].value_counts())
    
    # Guardar datos procesados
    df_processed.to_csv('data/processed/flights_processed.csv', index=False, encoding='utf-8')
    print("\n✓ Datos procesados guardados en data/processed/flights_processed.csv")
    