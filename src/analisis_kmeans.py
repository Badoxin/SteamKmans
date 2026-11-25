from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # guarda las graficas en archivos sin abrir ventanas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

K_FINAL = 4
sns.set_theme(style="whitegrid")

# Obtencion y exploracion del dataset
# Importar el dataset en Python utilizando read_csv()
df_original = pd.read_csv(DATOS / "steam_original.csv", encoding="utf-8")

# Mostrar los primeros 10 registros
print("Primeros 10 registros:")
print(df_original.head(10).to_string())

# Identificar las dimensiones, columnas y tipos de datos
print("\nDimensiones (filas, columnas):", df_original.shape)
print("\nColumnas y tipos de datos:")
print(df_original.dtypes)

# Obtener estadisticas descriptivas
print("\nEstadisticas descriptivas:")
print(df_original.describe().round(2).to_string())

# Identificar valores nulos y registros duplicados
print("\nValores nulos por columna:")
print(df_original.isna().sum()[df_original.isna().sum() > 0])
print("\nRegistros duplicados:", df_original.duplicated().sum())



# Crear al menos tres graficas exploratorias
# Grafica 1: distribucion del precio
plt.figure(figsize=(7, 4.5))
sns.histplot(df_original[df_original["price"] < 40]["price"], bins=40, color="#2a6f97")
plt.title("Distribucion del precio (juegos menores a 40 USD)")
plt.xlabel("Precio (USD)")
plt.ylabel("Juegos")
plt.tight_layout()
plt.savefig(FIGURAS / "explora_1_precio.png", dpi=130)
plt.close()

# Grafica 2: juegos por rango de propietarios
orden = df_original.assign(
    inicio=df_original["owners"].str.split("-").str[0].astype(float)
).groupby(["owners", "inicio"]).size().reset_index(name="juegos").sort_values("inicio")
plt.figure(figsize=(7, 4.5))
sns.barplot(data=orden, y="owners", x="juegos", color="#2a6f97")
plt.xscale("log")
plt.title("Juegos por rango de propietarios (escala log)")
plt.xlabel("Juegos")
plt.ylabel("Propietarios estimados")
plt.tight_layout()
plt.savefig(FIGURAS / "explora_2_propietarios.png", dpi=130)
plt.close()

# Grafica 3: correlacion entre variables numericas
cols_corr = ["price", "achievements", "positive_ratings", "negative_ratings",
             "average_playtime", "median_playtime"]
plt.figure(figsize=(7, 4.5))
sns.heatmap(df_original[cols_corr].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlacion entre variables numericas")
plt.tight_layout()
plt.savefig(FIGURAS / "explora_3_correlacion.png", dpi=130)
plt.close()



# Seleccionar al menos tres variables relevantes para el agrupamiento y justificar
VARIABLES = ["price", "rating_ratio", "total_ratings",
             "owners_mid", "average_playtime", "achievements"]



# Limpieza y preparacion de datos
df = df_original.copy()

# Tratamiento de valores nulos (developer y publisher no se usan en el modelo)
df["developer"] = df["developer"].fillna("Desconocido")
df["publisher"] = df["publisher"].fillna("Desconocido")

# Tratamiento de registros duplicados (en este dataset no hay)
df = df.drop_duplicates(subset="appid")

# Verificar y corregir tipos de datos
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year



# Corregir inconsistencia: owners es un rango en texto
limites = df["owners"].str.split("-", expand=True).astype(float)
df["owners_mid"] = (limites[0] + limites[1]) / 2

# Variables nuevas: total de resenas y porcentaje de resenas positivas
df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
df["rating_ratio"] = df["positive_ratings"] / df["total_ratings"] * 100



# Corregir inconsistencia: average_playtime = 0 significa que no hay dato de horas
# jugadas, por eso se eliminan esos registros
registros_inicio = len(df)
sin_playtime = int((df["average_playtime"] == 0).sum())
df = df[df["average_playtime"] > 0]

# Con menos de 20 resenas el porcentaje positivo no es confiable
con_pocas_resenas = int((df["total_ratings"] < 20).sum())
df = df[df["total_ratings"] >= 20].copy()

# Mostrar los resultados de la limpieza y preparacion
print("\nRegistros iniciales:", registros_inicio)
print("Eliminados por no tener horas jugadas:", sin_playtime)
print("Eliminados por tener menos de 20 resenas:", con_pocas_resenas)
print("Registros finales:", len(df))



# Identificar valores atipicos y justificar su tratamiento
print("\nValores atipicos por variable")
for v in VARIABLES:
    q1, q3 = df[v].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = int(((df[v] < q1 - 1.5 * iqr) | (df[v] > q3 + 1.5 * iqr)).sum())
    print(" ", v, ":", n, "(", round(n / len(df) * 100, 1), "%)")



# Aplicar transformaciones
VARS_LOG = ["price", "total_ratings", "owners_mid", "average_playtime", "achievements"]
X = df[VARIABLES].copy()
for v in VARS_LOG:
    X[v] = np.log1p(X[v])



# Implementar una tecnica de escalamiento
escalador = StandardScaler()
X_esc = escalador.fit_transform(X)



# Implementacion del algoritmo K-Means
# Evaluar cinco valores diferentes de K (de 2 a 6)
metricas = []
for k in range(2, 7):
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = m.fit_predict(X_esc)
    metricas.append({"K": k, "inercia": round(m.inertia_, 1),
                     "silhouette": round(silhouette_score(X_esc, etiquetas), 4)})
tabla_k = pd.DataFrame(metricas)
print("\nEvaluacion de valores de K:")
print(tabla_k.to_string(index=False))
tabla_k.to_csv(RESULTADOS / "metricas_k.csv", index=False)



# Grafica del metodo del codo y de la silueta para elegir K
fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
ejes[0].plot(tabla_k["K"], tabla_k["inercia"], marker="o", color="#2a6f97")
ejes[0].set_title("Metodo del codo (inercia)")
ejes[0].set_xlabel("K")
ejes[1].plot(tabla_k["K"], tabla_k["silhouette"], marker="o", color="#6a994e")
ejes[1].set_title("Coeficiente de silueta")
ejes[1].set_xlabel("K")
for eje in ejes:
    eje.axvline(K_FINAL, color="red", linestyle="--")
plt.tight_layout()
plt.savefig(FIGURAS / "kmeans_codo_silhouette.png", dpi=130)
plt.close()



# Seleccionar y justificar el numero de clusteres
# Entrenar el modelo final con KMeans y asignar un cluster con fit_predict()
modelo = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["cluster"] = modelo.fit_predict(X_esc)
print("\nIteraciones del modelo final:", modelo.n_iter_)



# Obtener los centroides (se convierten a unidades originales)
centros = escalador.inverse_transform(modelo.cluster_centers_)
centroides = pd.DataFrame(centros, columns=VARIABLES)
for v in VARS_LOG:
    centroides[v] = np.expm1(centroides[v])
centroides.insert(0, "cluster", range(K_FINAL))
print("\nCentroides (unidades originales):")
print(centroides.round(1).to_string(index=False))



# Identificar la cantidad de registros de cada cluster
print("\nRegistros por cluster:")
print(df["cluster"].value_counts().sort_index())



# Comparar dos configuraciones del modelo: K = 3 contra K = 4
modelo_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
etiq_3 = modelo_3.fit_predict(X_esc)
comparacion = pd.DataFrame([
    {"configuracion": "K = 3", "inercia": round(modelo_3.inertia_, 1),
     "silhouette": round(silhouette_score(X_esc, etiq_3), 4),
     "tamanos": sorted(np.bincount(etiq_3).tolist(), reverse=True)},
    {"configuracion": "K = 4", "inercia": round(modelo.inertia_, 1),
     "silhouette": round(silhouette_score(X_esc, df["cluster"]), 4),
     "tamanos": sorted(np.bincount(df["cluster"]).tolist(), reverse=True)},
])
print("\nComparacion de dos configuraciones:")
print(comparacion.to_string(index=False))
comparacion.to_csv(RESULTADOS / "comparacion_configuraciones.csv", index=False)
medianas_3 = df.assign(grupo3=etiq_3).groupby("grupo3")[VARIABLES].median().round(2)
medianas_3["juegos"] = np.bincount(etiq_3)
print("\nMedianas por grupo con K = 3:")
print(medianas_3.to_string())
medianas_3.to_csv(RESULTADOS / "medianas_k3.csv")



# Asignar un nombre descriptivo a cada cluster (reglas sobre las medianas)
medianas = df.groupby("cluster")[VARIABLES].median()
c_malos = int(medianas["rating_ratio"].idxmin())
resto = medianas.drop(index=c_malos)
c_premium = int(resto["total_ratings"].idxmax())
resto = resto.drop(index=c_premium)
c_baratos = int(resto["price"].idxmin())
c_indie = int([c for c in resto.index if c != c_baratos][0])
nombres = {c_premium: "Premium consolidados", c_indie: "Indies de nicho",
           c_baratos: "Baratos populares", c_malos: "Mal calificados"}
df["segmento"] = df["cluster"].map(nombres)
centroides["segmento"] = centroides["cluster"].map(nombres)




# Calcular estadisticas descriptivas por grupo y elaborar la tabla comparativa
tabla_comparativa = df.groupby("segmento").agg(
    juegos=("appid", "count"),
    precio_mediano=("price", "median"),
    porcentaje_positivo=("rating_ratio", "median"),
    resenas_medianas=("total_ratings", "median"),
    propietarios_medianos=("owners_mid", "median"),
    horas_jugadas_medianas=("average_playtime", lambda s: round(s.median() / 60, 1)),
    logros_medianos=("achievements", "median"),
).round(2)
tabla_comparativa["porcentaje_del_total"] = (
    tabla_comparativa["juegos"] / len(df) * 100).round(1)
print("\nTabla comparativa de los grupos (medianas):")
print(tabla_comparativa.to_string())
tabla_comparativa.to_csv(RESULTADOS / "tabla_comparativa.csv")
centroides.round(2).to_csv(RESULTADOS / "centroides.csv", index=False)



# Grafica para visualizar los agrupamientos
paleta = {"Premium consolidados": "#2a6f97", "Indies de nicho": "#6a994e",
          "Baratos populares": "#f4a259", "Mal calificados": "#bc4749"}
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=df, x="total_ratings", y="rating_ratio", hue="segmento",
                palette=paleta, s=14, alpha=0.6, linewidth=0)
plt.xscale("log")
plt.title("Segmentos de juegos de Steam")
plt.xlabel("Resenas totales (escala log)")
plt.ylabel("% de resenas positivas")
plt.legend(title="Segmento")
plt.tight_layout()
plt.savefig(FIGURAS / "resultados_segmentos.png", dpi=130)
plt.close()



# Guardar el dataset procesado con los clusters asignados
salida = ["appid", "name", "release_year", "price", "achievements", "positive_ratings",
          "negative_ratings", "total_ratings", "rating_ratio", "average_playtime",
          "owners", "owners_mid", "cluster", "segmento"]
df[salida].to_csv(DATOS / "steam_procesado.csv", index=False, encoding="utf-8")
print("\nDataset procesado guardado en datos/steam_procesado.csv")
