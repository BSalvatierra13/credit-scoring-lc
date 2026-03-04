# src/load_loans.py
import pandas as pd
from sqlalchemy import create_engine

# 1. Conexión a PostgreSQL (modo trust, sin contraseña)
engine = create_engine("postgresql://postgres@localhost/credit_scoring")

# 2. Columnas CSV y su mapeo a la tabla
csv_cols = [
    "id",               # mapea a loan_id
    "member_id",
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "dti",
    "revol_util",
    "earliest_cr_line",
    "open_acc",
    "loan_status"
]

# 3. Carga en chunks para no saturar memoria
chunksize = 100_000
csv_path = "data/raw/accepted_2007_to_2018Q4.csv"
# Modificación dentro de tu src/load_loans.py
for i, chunk in enumerate(pd.read_csv(
    csv_path,
    usecols=csv_cols + ["issue_d"], 
    parse_dates=["earliest_cr_line", "issue_d"],
    chunksize=chunksize,
    low_memory=False
)):
    # 1. PURGA DE DATOS CORRUPTOS: Eliminar las filas de resumen de LendingClub
    # Si no hay ID, no hay préstamo.
    chunk = chunk.dropna(subset=['id'])
    
    # 2. FILTRO TEMPORAL ESTRICTO (Desde 2016)
    chunk = chunk[chunk['issue_d'] >= '2016-01-01'].copy()
    
    if chunk.empty:
        continue

    chunk = chunk.drop(columns=['issue_d'])
    chunk = chunk.rename(columns={"id": "loan_id"})
    
    # 3. Inserción con bloque Try/Except para capturar el error exacto
    try:
        chunk.to_sql("loans", engine, if_exists="append", index=False, method='multi')
        print(f"Chunk {i+1} insertado con éxito.")
    except Exception as e:
        print(f"Error crítico en el chunk {i+1}: {e}")
        break # Detenemos la ejecución para no arruinar la tabla

print("Carga completa.")
