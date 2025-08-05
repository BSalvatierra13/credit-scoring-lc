-- ddl.sql: Esquema para Lending Club (simplificado)
CREATE TABLE IF NOT EXISTS loans (
  id SERIAL PRIMARY KEY,
  loan_id VARCHAR(50) UNIQUE,
  member_id INTEGER,
  loan_amnt NUMERIC(10,2),
  term VARCHAR(20),
  int_rate NUMERIC(5,2),
  grade CHAR(1),
  sub_grade VARCHAR(3),
  emp_length VARCHAR(20),
  home_ownership VARCHAR(20),
  annual_inc NUMERIC(12,2),
  issue_d DATE,
  loan_status VARCHAR(30)
  -- agrega más columnas según necesites
);
