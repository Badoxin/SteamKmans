# Segmentación de juegos de Steam con K-Means y dashboard interactivo

Proyecto integrador de la tercera unidad (minería de datos). Se agrupan **6,024 juegos de Steam** con el algoritmo de aprendizaje no supervisado **K-Means** según su precio, calificación, audiencia, horas jugadas y logros, y los resultados se presentan en un **dashboard interactivo hecho con Streamlit**.

**Integrantes:** _[escribe aquí tu nombre y el de tu equipo]_  
**Grupo:** _[escribe aquí tu grupo]_

## Problema

Steam publica miles de juegos y no es fácil saber qué tipos de juegos existen ni qué los distingue. El proyecto busca identificar grupos de juegos con características parecidas (precio, porcentaje de reseñas positivas, número de reseñas, propietarios estimados, tiempo jugado y logros) para entender qué hace diferente a cada grupo y qué recomendaciones se pueden derivar para desarrolladores y para la plataforma.

## Resultado principal

El modelo final usa **K = 4** y encuentra estos segmentos:

| Segmento | Juegos | Precio mediano | % reseñas positivas | Reseñas (mediana) | Propietarios (mediana) |
|---|---|---|---|---|---|
| Premium consolidados | 1,407 | $14.99 | 85.7% | 4,751 | 350,000 |
| Indies de nicho | 2,012 | $6.99 | 81.8% | 302 | 35,000 |
| Baratos populares | 1,230 | $0.79 | 80.7% | 1,222 | 350,000 |
| Mal calificados | 1,375 | $1.99 | 51.4% | 131 | 75,000 |

El coeficiente de silueta del modelo es 0.196, por lo que los grupos se traslapan en parte (ver limitaciones en el PDF).

## Herramientas

Python, pandas, NumPy, scikit-learn (KMeans, StandardScaler y silueta), Matplotlib, Seaborn, Plotly y Streamlit.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── datos/
│   ├── steam_original.csv        # dataset original (27,075 juegos, 18 variables)
│   └── steam_procesado.csv       # dataset limpio con el cluster y el segmento asignados
├── src/
│   └── analisis_kmeans.py        # exploración, limpieza, K-Means y análisis (partes 2 a 5)
├── dashboard/
│   └── app.py                    # dashboard de Streamlit (parte 6)
├── figuras/                      # gráficas generadas por el análisis
├── resultados/                   # métricas de K, centroides y tablas en CSV
└── docs/
    └── Documentacion_Proyecto.pdf
```

## Cómo ejecutarlo

1. Instalar las dependencias (se recomienda un entorno virtual):

```bash
pip install -r requirements.txt
```

2. Ejecutar el análisis desde la raíz del repositorio. Esto lee `datos/steam_original.csv` y vuelve a generar `datos/steam_procesado.csv`, las figuras y los resultados:

```bash
python src/analisis_kmeans.py
```

3. Abrir el dashboard (también desde la raíz del repositorio):

```bash
streamlit run dashboard/app.py
```

Se abre en el navegador, normalmente en `http://localhost:8501`. El dashboard lee `datos/steam_procesado.csv` y los archivos de `resultados/`, por lo que ya funciona sin volver a correr el análisis.

## Dashboard

- **Página 1, Análisis general:** 4 indicadores, distribución del precio, juegos por año y estadísticas descriptivas.
- **Página 2, Segmentación K-Means:** 4 indicadores, juegos por segmento, gráfica de dispersión, dos gráficas de comparación, tabla de estadísticas por segmento, descripción de los grupos, hallazgos y recomendaciones.
- **Filtros** (barra lateral, aplican a las dos páginas): segmento, rango de precio y año de lanzamiento.

## Dataset

- **Nombre:** Steam Store Games (datos de la Steam Store API y de SteamSpy).
- **Contenido:** 27,075 juegos lanzados entre 1997 y mayo de 2019, con 18 variables.
- **Copia utilizada:** archivo `steam_cleaned.csv` del repositorio público `zobairhas/Videogame_Trends_Analysis` en GitHub (se guardó aquí como `datos/steam_original.csv` sin modificarlo).

## Preparación de los datos (resumen)

- Se eliminaron los juegos sin horas jugadas registradas (`average_playtime = 0`, 20,905 registros) y los que tenían menos de 20 reseñas (146 registros), quedando 6,024 juegos.
- Se convirtió `owners` (un rango en texto) en un número con el punto medio del rango.
- Se crearon `total_ratings` y `rating_ratio` (% de reseñas positivas).
- Los valores atípicos se conservaron porque son juegos reales muy exitosos; se redujo su efecto con `log(1 + x)` y después se aplicó `StandardScaler`.
- Variables del modelo: `price`, `rating_ratio`, `total_ratings`, `owners_mid`, `average_playtime` y `achievements`.

## Limitaciones

- Los datos llegan hasta 2019.
- Los propietarios son rangos estimados por SteamSpy.
- Se excluyó la mayoría de los juegos del dataset por no tener horas jugadas registradas, así que los resultados describen juegos con jugadores registrados.
- La silueta es baja (0.196): los segmentos se traslapan y no son grupos perfectamente separados.
