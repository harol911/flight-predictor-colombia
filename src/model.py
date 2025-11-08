"""
Modelo de Machine Learning para predicción de mejores ofertas de vuelos.

Este sistema utiliza la API de Amadeus para obtener datos en tiempo real de vuelos
desde el Aeropuerto Internacional Benito Juárez (MEX) en Ciudad de México hacia el
Aeropuerto Internacional José María Córdova (MDE) en Medellín, Colombia.

El modelo analiza múltiples variables (precio, aerolínea, escalas, duración, días
de anticipación) para identificar y recomendar las mejores opciones de vuelo
disponibles durante todo el año.

Autor: Harol Paz
Universidad Politécnica de Santa Rosa Jáuregui
Ingeniería en Robótica Computacional
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class FlightPricePredictor:
    def __init__(self):
        self.model = None
        self.best_model = None
        self.feature_importance = None
        self.evaluation_results = {}
    
    def load_processed_data(self, filepath='data/processed/flights_processed.csv'):
        """Cargar datos procesados"""
        df = pd.read_csv(filepath)
        print(f"✓ Datos cargados: {len(df)} registros")
        return df
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """Entrenar y comparar múltiples modelos"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\n" + "="*70)
        print("COMPARACIÓN DE MODELOS DE MACHINE LEARNING")
        print("="*70)
        
        for name, model in models.items():
            print(f"\n🔄 Entrenando {name}...")
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas en entrenamiento
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_pred_train)
            
            # Métricas en prueba
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'test_mape': test_mape,
                'cv_rmse': cv_rmse,
                'y_pred': y_pred_test
            }
            
            print(f"\n📊 Resultados de {name}:")
            print(f"   Entrenamiento:")
            print(f"      - RMSE: ${train_rmse:.2f} MXN")
            print(f"      - R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
            print(f"   Prueba:")
            print(f"      - RMSE: ${test_rmse:.2f} MXN")
            print(f"      - MAE: ${test_mae:.2f} MXN")
            print(f"      - R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
            print(f"      - MAPE: {test_mape:.2f}%")
            print(f"   Validación Cruzada (5-fold):")
            print(f"      - RMSE promedio: ${cv_rmse:.2f} MXN")
        
        # Seleccionar mejor modelo (menor RMSE en prueba)
        best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
        self.best_model = results[best_model_name]['model']
        self.evaluation_results = results
        
        print(f"\n" + "="*70)
        print(f"✅ MEJOR MODELO: {best_model_name}")
        print(f"   RMSE: ${results[best_model_name]['test_rmse']:.2f} MXN")
        print(f"   R²: {results[best_model_name]['test_r2']:.4f}")
        print("="*70)
        
        return results, best_model_name
    
    def analyze_feature_importance(self, X, feature_names):
        """Analizar importancia de features"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance
            
            # Visualizar top 10
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance.head(10), x='importance', y='feature', palette='viridis')
            plt.title('Top 10 Features Más Importantes', fontsize=16, fontweight='bold')
            plt.xlabel('Importancia', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n✓ Gráfico de importancia guardado en: models/feature_importance.png")
            
            return importance
        
        return None
    
    def plot_predictions_vs_actual(self, y_test, best_model_name):
        """Graficar predicciones vs valores reales"""
        y_pred = self.evaluation_results[best_model_name]['y_pred']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
        
        # Línea de predicción perfecta
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        plt.xlabel('Precio Real (MXN)', fontsize=12)
        plt.ylabel('Precio Predicho (MXN)', fontsize=12)
        plt.title(f'Predicciones vs Valores Reales - {best_model_name}', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de predicciones guardado en: models/predictions_vs_actual.png")
    
    def plot_residuals(self, y_test, best_model_name):
        """Graficar residuales"""
        y_pred = self.evaluation_results[best_model_name]['y_pred']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuales vs Predicciones
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Precio Predicho (MXN)', fontsize=12)
        axes[0].set_ylabel('Residual (MXN)', fontsize=12)
        axes[0].set_title('Residuales vs Predicciones', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Distribución de residuales
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residual (MXN)', fontsize=12)
        axes[1].set_ylabel('Frecuencia', fontsize=12)
        axes[1].set_title('Distribución de Residuales', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('models/residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de residuales guardado en: models/residuals_analysis.png")
    
    def compare_models_visualization(self):
        """Visualizar comparación de modelos"""
        model_names = list(self.evaluation_results.keys())
        test_rmse = [self.evaluation_results[m]['test_rmse'] for m in model_names]
        test_r2 = [self.evaluation_results[m]['test_r2'] for m in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSE
        axes[0].bar(model_names, test_rmse, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
        axes[0].set_ylabel('RMSE (MXN)', fontsize=12)
        axes[0].set_title('Comparación de RMSE (menor es mejor)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(test_rmse):
            axes[0].text(i, v + 50, f'${v:.0f}', ha='center', fontweight='bold')
        
        # R²
        axes[1].bar(model_names, test_r2, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
        axes[1].set_ylabel('R² Score', fontsize=12)
        axes[1].set_title('Comparación de R² (mayor es mejor)', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(test_r2):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de comparación guardado en: models/models_comparison.png")
    
    def save_model(self, filepath='models/flight_predictor.joblib'):
        """Guardar modelo"""
        joblib.dump(self.best_model, filepath)
        print(f"\n✓ Modelo guardado en: {filepath}")
    
    def predict_best_flights(self, df, X, top_n=10):
        """Predecir y recomendar mejores vuelos"""
        df = df.copy()
        df['precio_predicho'] = self.best_model.predict(X)
        
        # Calcular "valor" (precio predicho - precio real)
        # Si precio_predicho > precio_real = buena oferta
        df['ahorro_potencial'] = df['precio_predicho'] - df['precio']
        
        # Ordenar por mejor valor (mayor ahorro)
        best_deals = df.nlargest(top_n, 'ahorro_potencial')
        
        return best_deals[['origen', 'destino', 'aerolinea', 'fecha_salida', 
                          'precio', 'precio_predicho', 'ahorro_potencial', 
                          'escalas', 'duracion_minutos', 'dias_anticipacion']]

if __name__ == "__main__":
    from preprocessing import FlightPreprocessor
    
    print("="*70)
    print("ENTRENAMIENTO Y EVALUACIÓN DEL MODELO DE PREDICCIÓN DE VUELOS")
    print("="*70)
    
    # Cargar y preparar datos
    preprocessor = FlightPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.feature_engineering(df)
    df = preprocessor.create_target_variable(df)
    X, y, df_processed = preprocessor.prepare_for_ml(df)
    
    # Split datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 División de datos:")
    print(f"   ✓ Datos de entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   ✓ Datos de prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Entrenar y comparar modelos
    predictor = FlightPricePredictor()
    results, best_model_name = predictor.train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Analizar importancia de features
    print("\n" + "="*70)
    print("ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*70)
    importance = predictor.analyze_feature_importance(X, X.columns)
    if importance is not None:
        print("\nTop 10 Features Más Importantes:")
        print(importance.head(10).to_string(index=False))
    
    # Crear visualizaciones
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70)
    predictor.plot_predictions_vs_actual(y_test, best_model_name)
    predictor.plot_residuals(y_test, best_model_name)
    predictor.compare_models_visualization()
    
    # Guardar modelo
    predictor.save_model()
    
    # Predecir mejores vuelos
    print("\n" + "="*70)
    print("TOP 10 MEJORES OFERTAS DETECTADAS")
    print("="*70)
    best_flights = predictor.predict_best_flights(df_processed, X, top_n=10)
    print(best_flights.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\nArchivos generados:")
    print("  📊 models/feature_importance.png")
    print("  📊 models/predictions_vs_actual.png")
    print("  📊 models/residuals_analysis.png")
    print("  📊 models/models_comparison.png")
    print("  🤖 models/flight_predictor.joblib")
    print("\n¡Modelo listo para producción! 🚀")