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
WITH base as ( --Filter the not null loan_status
    SELECT *
    FROM loans
    WHERE loan_status IS NOT NULL
),
labeled as ( --Creating the target column
    SELECT  *,
            CASE 
                WHEN loan_status IN ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid', 'In Grace Period', 'Current') THEN 0
                WHEN loan_status IN ('Late (16-30 days)', 'Late (31-120 days)', 'Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off') THEN 1
            END AS default_status
    FROM base
),
typed as ( -- Type casting and basic cleaning of variables:
           -- Convert term to number of months (integer)
           -- Convert int_rate, dti, revol_util from percentage strings to decimals
           -- Standardize emp_length: extract numeric years, map '< 1 year' to 0, '10+ years' to 10, NULL to 'Unknown'
           -- Create binary flag emp_length_missing to indicate missing values
           -- Cast earliest_cr_line to date and open_acc to integer
           -- Keep default_status from previous step
    SELECT loan_id::INTEGER,
           member_id::INTEGER,
           loan_amnt::NUMERIC,
           REGEXP_REPLACE(term, '[^0-9]', '', 'g')::int AS term_months,
           ROUND(int_rate/100,3) :: NUMERIC AS int_rate,
           grade,
           sub_grade,
           CASE 
               WHEN emp_length IS NULL THEN 'Unknown' -- We have many of this, so we create a new category
               WHEN emp_length LIKE '%10+' THEN '10'
               WHEN emp_length LIKE '%< 1' THEN '0'
               ELSE REGEXP_REPLACE(emp_length, '[^0-9]', '', 'g')
           END AS emp_length_years_txt,
           home_ownership,
           annual_inc::NUMERIC,
           ROUND(dti/100,3) :: NUMERIC AS dti,
           ROUND(revol_util/100,3) :: NUMERIC AS revol_util,
           CAST(earliest_cr_line AS DATE) AS earliest_cr_line,
           open_acc::INTEGER,
           default_status
    FROM labeled
),
med as (-- Calculate median values for dti and revol_util by grade
    SELECT grade,
              PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dti) AS dti_median,
              PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revol_util) AS revol_median 
    FROM typed
    GROUP BY grade
),
imput as ( -- Imputation of missing values for dti and revol_util
           -- Missing values are replaced with the median (50th percentile) 
           -- of the corresponding grade category
    SELECT
        typed.*,
        COALESCE(typed.dti, med.dti_median) AS dti_imp,
        COALESCE(typed.revol_util, med.revol_median) AS revol_imp
    FROM typed
    LEFT JOIN med ON typed.grade = med.grade
), 
feat as ( -- Selection of the relevant features for modeling
          -- Avoiding member_id because it has many NULL values
          -- Extracting year from earliest_cr_line
    SELECT 
        loan_id as id,
        loan_amnt,
        term_months,
        int_rate,
        grade,
        sub_grade,
        emp_length_years_txt,
        home_ownership,
        annual_inc,
        dti_imp as dti,
        revol_imp as revol_util,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, earliest_cr_line)) AS earliest_cr_line,
        open_acc,
        default_status
    FROM imput
), 
final as ( -- Final selection of features for the model
           -- We delete all the NULL values
    SELECT *
    FROM feat
    WHERE
        loan_amnt IS NOT NULL AND
        term_months IS NOT NULL AND
        int_rate IS NOT NULL AND
        grade IS NOT NULL AND
        sub_grade IS NOT NULL AND
        home_ownership IS NOT NULL AND
        annual_inc IS NOT NULL AND
        earliest_cr_line IS NOT NULL AND
        open_acc IS NOT NULL
)

SELECT *
FROM final;