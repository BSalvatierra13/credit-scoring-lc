-- sql/ddl.sql: Esquema para Lending Club (con variables iniciales)
CREATE TABLE IF NOT EXISTS loans (
  id SERIAL PRIMARY KEY,
  loan_id      VARCHAR(50)  UNIQUE,
  member_id    INTEGER,
  loan_amnt    NUMERIC(10,2),
  term         VARCHAR(20),
  int_rate     NUMERIC(5,2),
  grade        CHAR(1),
  sub_grade    VARCHAR(3),
  emp_length   VARCHAR(20),
  home_ownership VARCHAR(20),

  -- variables seleccionadas
  annual_inc        NUMERIC(12,2),  -- ingreso anual
  dti               NUMERIC(5,2),   -- debt-to-income ratio
  revol_util        NUMERIC(5,2),   -- % de línea revolvente usada
  earliest_cr_line  DATE,           -- fecha de primera línea de crédito
  open_acc          INTEGER         -- número de cuentas abiertas

  -- más columnas las añadiremos luego
);