import pandas as pd
from datetime import datetime, timedelta
import random

class FlightScraper:
    def __init__(self):
        self.flights_data = []
    
    def generate_sample_data(self, num_samples=1000):
        """
        Generar datos de muestra para desarrollo
        Simula vuelos desde Ciudad de México a Medellín
        """
        airlines = ['Aeroméxico', 'Volaris', 'VivaAerobus', 'Avianca', 'LATAM', 'Copa Airlines']
        origins = ['MEX']  # Solo Ciudad de México
        destinations = ['MDE']  # Solo Medellín
        
        data = []
        
        print(f"Generando {num_samples} registros de vuelos MEX → MDE...")
        
        for i in range(num_samples):
            base_date = datetime.now() + timedelta(days=random.randint(1, 180))
            duration = random.randint(180, 720)
            
            flight = {
                'fecha_busqueda': datetime.now().strftime('%Y-%m-%d'),
                'origen': random.choice(origins),
                'destino': random.choice(destinations),
                'aerolinea': random.choice(airlines),
                'fecha_salida': base_date.strftime('%Y-%m-%d'),
                'hora_salida': f"{random.randint(0, 23):02d}:{random.choice(['00', '30'])}",
                'fecha_llegada': (base_date + timedelta(minutes=duration)).strftime('%Y-%m-%d'),
                'hora_llegada': f"{random.randint(0, 23):02d}:{random.choice(['00', '30'])}",
                'precio': random.randint(4663, 15000),
                'moneda': 'MXN',
                'escalas': random.choice([0, 0, 0, 1, 1, 2]),  # Más vuelos directos
                'duracion_minutos': duration,
                'asientos_disponibles': random.randint(1, 50),
                'clase': random.choice(['Economica', 'Economica', 'Premium', 'Ejecutiva']),
                'equipaje_incluido': random.choice([True, False]),
                'dias_anticipacion': random.randint(1, 180)
            }
            
            data.append(flight)
        
        df = pd.DataFrame(data)
        return df
    
    def save_data(self, df, filename='data/raw/flights_data.csv'):
        """Guardar datos en CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Datos guardados en {filename}")

if __name__ == "__main__":
    print("="*60)
    print("SCRAPER DE VUELOS CDMX → MEDELLÍN")
    print("="*60)
    
    scraper = FlightScraper()
    
    # Generar datos de muestra
    df = scraper.generate_sample_data(1000)
    
    # Guardar datos
    scraper.save_data(df)
    
    print(f"\n✓ Total de vuelos generados: {len(df)}")
    print("\nPrimeros 5 registros:")
    print(df.head().to_string())
    print("\nEstadísticas básicas:")
    print(f"  💰 Precio promedio: ${df['precio'].mean():.2f} MXN")
    print(f"  💵 Precio mínimo: ${df['precio'].min():.2f} MXN")
    print(f"  💸 Precio máximo: ${df['precio'].max():.2f} MXN")
    print(f"  ⏱️  Duración promedio: {df['duracion_minutos'].mean():.0f} minutos")
    print(f"  🔄 Escalas: {df['escalas'].value_counts().to_dict()}")