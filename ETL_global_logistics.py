import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import os

# Load Data
df = pd.read_excel("data/International_LPI_from_2007_to_2023_0.xlsx")

# Data Cleaning
columns_to_keep = [
    'Economy', 'LPI Score', 'Customs Score', 'Infrastructure Score', 
    'International Shipments Score', 'Logistics Competence and Quality Score', 
    'Timeliness Score', 'Tracking and Tracing Score'
]

df_cleaned = df[columns_to_keep]

# Convert to numeric
numeric_columns = columns_to_keep[1:]  # All but the first column
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Data Analysis
df_cleaned.describe()
df_cleaned.info()
print("Cantidad de nulos: ", df_cleaned.isnull().sum())

# Calculate correlation Infrastructure Score and International Shipments Score
corr_infra_shipment = df_cleaned[['Infrastructure Score', 'International Shipments Score']].dropna().corr()
print("Correlación entre Puntaje de Infraestructura y Puntaje Envíos Internacionales:\n", corr_infra_shipment)

# Regression Analysis
x1 = df_cleaned['Infrastructure Score']
y1 = df_cleaned['International Shipments Score']
x1 = sm.add_constant(x1)  # Añadir una constante para el intercepto
model = sm.OLS(y1, x1).fit()
print(model.summary())

# Calculate correlation between Customs Score and Timeliness Score
corr_customs_timeliness = df_cleaned[['Customs Score', 'Timeliness Score']].dropna().corr()
print("Correlación entre Customs Score y Timeliness Score:\n", corr_customs_timeliness)

# Regression Analysis
x2 = df_cleaned['Customs Score']
y2 = df_cleaned['Timeliness Score']
x2 = sm.add_constant(x2)
model_customs = sm.OLS(y2, x2).fit()
print(model_customs.summary())

# Save cleaned data
df_cleaned.to_excel("cleaned_data.xlsx", index=False)

# Strengths and Weaknesses per economy
# Rank for each KPI (menor valor = mejor ranking)
df_cleaned.loc[:,'Infrastructure Rank'] = df_cleaned['Infrastructure Score'].rank(ascending=False)
df_cleaned.loc[:,'Customs Rank'] = df_cleaned['Customs Score'].rank(ascending=False)
df_cleaned.loc[:,'Shipments Rank'] = df_cleaned['International Shipments Score'].rank(ascending=False)
df_cleaned.loc[:,'Logistics Rank'] = df_cleaned['Logistics Competence and Quality Score'].rank(ascending=False)
df_cleaned.loc[:,'Timeliness Rank'] = df_cleaned['Timeliness Score'].rank(ascending=False)
df_cleaned.loc[:,'Tracking Rank'] = df_cleaned['Tracking and Tracing Score'].rank(ascending=False)

# Overall Rank
df_cleaned['Overall Rank'] = df_cleaned[['Infrastructure Rank', 'Customs Rank', 'Shipments Rank', 'Logistics Rank', 'Timeliness Rank', 'Tracking Rank']].mean(axis=1).rank(ascending=True, method='min')

# Verificar algunas filas para confirmar los rangos
print(df_cleaned[['Economy', 'Infrastructure Rank', 'Customs Rank', 'Shipments Rank', 'Logistics Rank', 'Timeliness Rank', 'Tracking Rank']].head())

df_cleaned['Total Score'] = df_cleaned[['Infrastructure Score', 'Customs Score', 'International Shipments Score', 'Logistics Competence and Quality Score', 'Timeliness Score', 'Tracking and Tracing Score']].mean(axis=1)
df_cleaned = df_cleaned.sort_values(['Overall Rank', 'Total Score'], ascending=[True, False])

print(df_cleaned[['Economy', 'Overall Rank', 'Total Score']])

top_countries = df_cleaned.nlargest(10, 'Total Score')[['Economy', 'Total Score', 'Overall Rank']]
bottom_countries = df_cleaned.nsmallest(10, 'Total Score')[['Economy', 'Total Score', 'Overall Rank']]

print("Top 10 Países:\n", top_countries)
print("Bottom 10 Países:\n", bottom_countries)

# Guardar los datos procesados
df_cleaned.to_csv("cleaned_data_with_rankings.csv", index=False)


file_path = os.path.abspath("cleaned_data_with_rankings.csv")
print(file_path)
