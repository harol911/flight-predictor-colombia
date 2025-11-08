from amadeus import Client, ResponseError
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

# Cargar variables de entorno
load_dotenv()

class RealFlightScraper:
    def __init__(self):
        # Inicializar cliente de Amadeus
        api_key = os.getenv('AMADEUS_API_KEY')
        api_secret = os.getenv('AMADEUS_API_SECRET')
        
        if not api_key or not api_secret:
            raise Exception("❌ Error: No se encontraron las credenciales en el archivo .env")
        
        self.amadeus = Client(
            client_id=api_key,
            client_secret=api_secret
        )
        self.flights_data = []
        print("✓ Cliente de Amadeus inicializado correctamente")
    
    def search_flights(self, origin, destination, departure_date, adults=1):
        """
        Buscar vuelos reales usando Amadeus API
        """
        try:
            response = self.amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin,
                destinationLocationCode=destination,
                departureDate=departure_date,
                adults=adults,
                max=50  # Máximo de resultados
            )
            
            return response.data
        
        except ResponseError as error:
            print(f"❌ Error en la búsqueda: {error}")
            return []
    
    def parse_flight_data(self, flight_offer):
        """
        Extraer información relevante de cada vuelo
        """
        try:
            # Información básica
            precio = float(flight_offer['price']['total'])
            moneda = flight_offer['price']['currency']
            
            # Primer segmento (ida)
            itinerary = flight_offer['itineraries'][0]
            segments = itinerary['segments']
            
            primer_segmento = segments[0]
            ultimo_segmento = segments[-1]
            
            # Extraer datos
            origen = primer_segmento['departure']['iataCode']
            destino = ultimo_segmento['arrival']['iataCode']
            
            fecha_salida = primer_segmento['departure']['at'][:10]
            hora_salida = primer_segmento['departure']['at'][11:16]
            
            fecha_llegada = ultimo_segmento['arrival']['at'][:10]
            hora_llegada = ultimo_segmento['arrival']['at'][11:16]
            
            # Duración en minutos
            duracion_str = itinerary['duration'][2:]  # Quitar 'PT'
            duracion_minutos = self._parse_duration(duracion_str)
            
            # Número de escalas
            escalas = len(segments) - 1
            
            # Aerolínea
            aerolinea_code = primer_segmento['carrierCode']
            
            # Calcular días de anticipación
            fecha_busqueda = datetime.now()
            fecha_salida_dt = datetime.strptime(fecha_salida, '%Y-%m-%d')
            dias_anticipacion = (fecha_salida_dt - fecha_busqueda).days
            
            # Calcular precio con impuestos mexicanos
            if moneda == 'EUR':
                # Convertir EUR a MXN (aproximado)
                tasa_cambio = 21.5  # 1 EUR ≈ 21.5 MXN
                precio_mxn = precio * tasa_cambio
            elif moneda == 'USD':
                # Convertir USD a MXN (aproximado)
                tasa_cambio = 18.0  # 1 USD ≈ 18 MXN
                precio_mxn = precio * tasa_cambio
            else:
                precio_mxn = precio
            
            # Calcular impuestos y cargos (aproximadamente 30% adicional en México)
            iva = precio_mxn * 0.16  # IVA 16%
            tua = 650  # Tarifa de Uso de Aeropuerto internacional
            cargos = precio_mxn * 0.10  # Otros cargos (combustible, servicio, etc.)
            
            precio_base = precio_mxn
            precio_total = precio_mxn + iva + tua + cargos
            
            # FILTRO: Solo vuelos de $8,000 MXN o más
            if precio_total < 8000:
                return None
            
            flight_data = {
                'fecha_busqueda': fecha_busqueda.strftime('%Y-%m-%d'),
                'origen': origen,
                'destino': destino,
                'aerolinea': aerolinea_code,
                'fecha_salida': fecha_salida,
                'hora_salida': hora_salida,
                'fecha_llegada': fecha_llegada,
                'hora_llegada': hora_llegada,
                'precio_base': round(precio_base, 2),
                'iva': round(iva, 2),
                'tua': tua,
                'otros_cargos': round(cargos, 2),
                'precio': round(precio_total, 2),
                'moneda': 'MXN',
                'moneda_original': moneda,
                'precio_original': round(precio, 2),
                'escalas': escalas,
                'duracion_minutos': duracion_minutos,
                'asientos_disponibles': flight_offer.get('numberOfBookableSeats', 9),
                'clase': 'Economica',
                'equipaje_incluido': True,
                'dias_anticipacion': dias_anticipacion
            }
            
            return flight_data
        
        except Exception as e:
            print(f"❌ Error parseando vuelo: {e}")
            return None
    
    def _parse_duration(self, duration_str):
        """
        Convertir duración PT4H30M a minutos
        """
        hours = 0
        minutes = 0
        
        if 'H' in duration_str:
            hours = int(duration_str.split('H')[0])
            duration_str = duration_str.split('H')[1]
        
        if 'M' in duration_str:
            minutes = int(duration_str.replace('M', ''))
        
        return hours * 60 + minutes
    
    def scrape_multiple_routes(self, routes, date_range_days=30):
        """
        Buscar vuelos para múltiples rutas y fechas
        
        routes: lista de tuplas [(origen, destino), ...]
        """
        all_flights = []
        
        print("\n" + "="*60)
        print("INICIANDO BÚSQUEDA DE VUELOS REALES CON AMADEUS API")
        print("="*60)
        
        for origin, destination in routes:
            print(f"\n🔍 Buscando: {origin} → {destination}")
            
            # Buscar para varias fechas
            for days_ahead in range(7, date_range_days, 7):  # Cada 7 días
                departure_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                
                print(f"  📅 Fecha: {departure_date}", end=" ")
                
                # Buscar vuelos
                flight_offers = self.search_flights(origin, destination, departure_date)
                
                if flight_offers:
                    vuelos_filtrados = 0
                    # Parsear cada vuelo
                    for offer in flight_offers:
                        flight_data = self.parse_flight_data(offer)
                        if flight_data:
                            all_flights.append(flight_data)
                            vuelos_filtrados += 1
                    
                    print(f"✓ {vuelos_filtrados} vuelos encontrados")
                else:
                    print("✗ Sin resultados")
                
                # Esperar para no saturar la API
                time.sleep(1)
        
        return pd.DataFrame(all_flights)
    
    def save_data(self, df, filename='data/raw/flights_data_real.csv'):
        """Guardar datos reales en CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✓ {len(df)} vuelos reales guardados en {filename}")

if __name__ == "__main__":
    print("="*60)
    print("FLIGHT PREDICTOR COLOMBIA - SCRAPER DE DATOS REALES")
    print("="*60)
    
    try:
        scraper = RealFlightScraper()
        
        # Definir rutas desde México a Colombia
        routes = [
            ('MEX', 'MDE'),  # Ciudad de México → Medellín
        ]
        
        print(f"\n📍 Rutas a buscar: {len(routes)}")
        for origin, dest in routes:
            print(f"   • {origin} → {dest}")
        
        # Buscar vuelos
        df = scraper.scrape_multiple_routes(routes, date_range_days=28)
        
        if len(df) > 0:
            # Guardar datos
            scraper.save_data(df)
            
            print("\n" + "="*60)
            print("RESUMEN DE DATOS RECOPILADOS")
            print("="*60)
            print(f"Total de vuelos: {len(df)}")
            print(f"\nPrimeros 5 registros:")
            print(df[['aerolinea', 'fecha_salida', 'precio_base', 'precio', 'escalas', 'duracion_minutos']].head().to_string())
            print(f"\nEstadísticas de PRECIOS CON IMPUESTOS:")
            print(f"  💰 Precio promedio TOTAL: ${df['precio'].mean():.2f} MXN (${df['precio'].mean()/18:.2f} USD)")
            print(f"  💵 Precio mínimo TOTAL: ${df['precio'].min():.2f} MXN (${df['precio'].min()/18:.2f} USD)")
            print(f"  💸 Precio máximo TOTAL: ${df['precio'].max():.2f} MXN (${df['precio'].max()/18:.2f} USD)")
            print(f"\n  📊 Desglose promedio:")
            print(f"     - Precio base: ${df['precio_base'].mean():.2f} MXN")
            print(f"     - IVA (16%): ${df['iva'].mean():.2f} MXN")
            print(f"     - TUA: ${df['tua'].mean():.2f} MXN")
            print(f"     - Otros cargos: ${df['otros_cargos'].mean():.2f} MXN")
            print(f"\n  ⏱️  Duración promedio: {df['duracion_minutos'].mean():.0f} minutos")
            print(f"  🔄 Escalas promedio: {df['escalas'].mean():.1f}")
            print(f"\nDistribución por aerolínea:")
            print(df['aerolinea'].value_counts())
            
            print("\n✅ ¡Datos reales con impuestos obtenidos exitosamente!")
            print("\nSiguientes pasos:")
            print("  1. copy data\\raw\\flights_data_real.csv data\\raw\\flights_data.csv")
            print("  2. python src/preprocessing.py")
            print("  3. python src/model.py")
        else:
            print("\n❌ No se encontraron vuelos.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        