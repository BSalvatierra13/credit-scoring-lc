-- Let´s see how many loans we have in each status
SELECT loan_status, COUNT(*)
FROM loans
GROUP BY loan_status

-- Defining the default loan status and counting the number of loans in each status.
-- And dropping the null values
SELECT 
    CASE 
        WHEN loan_status IN ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid', 'In Grace Period', 'Current') THEN 0
        WHEN loan_status IN ('Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off') THEN 1
    END AS default_status,
    COUNT(*) AS status_count
FROM loans
WHERE loan_status IS NOT NULL
GROUP BY default_status;

-- Let´s see the NULL values in each column
SELECT 
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS id_nulls,
    SUM(CASE WHEN member_id IS NULL THEN 1 ELSE 0 END) AS member_id_nulls,
    SUM(CASE WHEN loan_amnt IS NULL THEN 1 ELSE 0 END) AS loan_amnt_nulls,
    SUM(CASE WHEN term IS NULL THEN 1 ELSE 0 END) AS term_nulls,
    SUM(CASE WHEN int_rate IS NULL THEN 1 ELSE 0 END) AS int_rate_nulls,
    SUM(CASE WHEN grade IS NULL THEN 1 ELSE 0 END) AS grade_nulls,
    SUM(CASE WHEN sub_grade IS NULL THEN 1 ELSE 0 END) AS sub_grade_nulls,
    SUM(CASE WHEN emp_length IS NULL THEN 1 ELSE 0 END) AS emp_length_nulls,
    SUM(CASE WHEN home_ownership IS NULL THEN 1 ELSE 0 END) AS home_ownership_nulls,
    SUM(CASE WHEN annual_inc IS NULL THEN 1 ELSE 0 END) AS annual_inc_nulls,
    SUM(CASE WHEN dti IS NULL THEN 1 ELSE 0 END) AS dti_nulls,
    SUM(CASE WHEN revol_util IS NULL THEN 1 ELSE 0 END) AS revol_util_nulls,
    SUM(CASE WHEN earliest_cr_line IS NULL THEN 1 ELSE 0 END) AS earliest_cr_line_nulls,
    SUM(CASE WHEN open_acc IS NULL THEN 1 ELSE 0 END) AS open_acc_nulls,
    SUM(CASE WHEN loan_status IS NULL THEN 1 ELSE 0 END) AS loan_status_nulls
FROM loans;

-- Now we start the actual database creation
CREATE TABLE loans_cleaned AS
WITH base AS ( 
    -- Filter the not null loan_status
    SELECT *
    FROM loans
    WHERE loan_status IS NOT NULL
),
labeled AS ( 
    -- Creating the target column
    SELECT  *,
            CASE 
                WHEN loan_status IN ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid', 'In Grace Period', 'Current') THEN 0
                WHEN loan_status IN ('Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off') THEN 1
            END AS default_status
    FROM base
),
typed AS ( 
    -- Type casting and basic cleaning of variables
    SELECT loan_id::INTEGER AS id,
           loan_amnt::NUMERIC,
           REGEXP_REPLACE(term, '[^0-9]', '', 'g')::INTEGER AS term_months,
           ROUND(int_rate/100, 3)::NUMERIC AS int_rate,
           grade,
           sub_grade,
           -- CORRECCIÓN OBLIGATORIA: Forzamos tipo INTEGER, usamos NULL en vez de 'Unknown'
           CASE 
               WHEN emp_length IS NULL THEN NULL
               WHEN emp_length LIKE '%10+' THEN 10
               WHEN emp_length LIKE '%< 1' THEN 0
               ELSE REGEXP_REPLACE(emp_length, '[^0-9]', '', 'g')::INTEGER
           END AS emp_length_years,
           home_ownership,
           annual_inc::NUMERIC,
           ROUND(dti/100, 3)::NUMERIC AS dti,
           ROUND(revol_util/100, 3)::NUMERIC AS revol_util,
           CAST(earliest_cr_line AS DATE) AS earliest_cr_line,
           open_acc::INTEGER,
           default_status
    FROM labeled
),
global_stats AS (
    -- Cálculo matemático del percentil 99.9 sobre toda la población de ingresos
    SELECT PERCENTILE_CONT(0.999) WITHIN GROUP (ORDER BY annual_inc) AS inc_p999
    FROM typed
),
med AS (
    -- Calculate median values for dti and revol_util by grade
    SELECT grade,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dti) AS dti_median,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revol_util) AS revol_median 
    FROM typed
    GROUP BY grade
),
imput AS ( 
    -- Imputation of missing values and joining global statistics
    SELECT typed.*,
           COALESCE(typed.dti, med.dti_median) AS dti_imp,
           COALESCE(typed.revol_util, med.revol_median) AS revol_imp,
           global_stats.inc_p999
    FROM typed
    LEFT JOIN med ON typed.grade = med.grade
    CROSS JOIN global_stats
), 
feat AS ( 
    -- Selection of the relevant features
    SELECT id,
           loan_amnt,
           term_months,
           int_rate,
           grade,
           sub_grade,
           emp_length_years,
           home_ownership,
           annual_inc,
           dti_imp AS dti,
           revol_imp AS revol_util,
           -- CORRECCIÓN OBLIGATORIA: Marco de referencia estático. Solo extraemos el año de inicio.
           EXTRACT(YEAR FROM earliest_cr_line) AS earliest_cr_line_year,
           open_acc,
           default_status,
           inc_p999
    FROM imput
), 
final AS ( 
    -- Final selection and application of domain filters
    SELECT 
        id, loan_amnt, term_months, int_rate, grade, sub_grade, 
        emp_length_years, home_ownership, annual_inc, dti, 
        revol_util, earliest_cr_line_year, open_acc, default_status
    FROM feat
    WHERE
        loan_amnt IS NOT NULL AND
        term_months IS NOT NULL AND
        int_rate IS NOT NULL AND
        grade IS NOT NULL AND
        sub_grade IS NOT NULL AND
        home_ownership IS NOT NULL AND
        annual_inc IS NOT NULL AND
        earliest_cr_line_year IS NOT NULL AND
        open_acc IS NOT NULL AND
        emp_length_years IS NOT NULL AND
        -- TUS NUEVAS REGLAS DE FILTRADO:
        dti >= 0 AND 
        annual_inc <= inc_p999
)
SELECT *
FROM final;