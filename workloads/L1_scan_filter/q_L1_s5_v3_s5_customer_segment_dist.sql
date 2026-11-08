SELECT
    c_mktsegment,
    c_nationkey,
    COUNT(*) AS customer_cnt,
    SUM(c_acctbal) AS sum_acctbal,
    AVG(c_acctbal) AS avg_acctbal,
    MIN(c_acctbal) AS min_acctbal,
    MAX(c_acctbal) AS max_acctbal,
    SUM(CASE
        WHEN c_acctbal >= 8000 THEN c_acctbal * 1.05
        WHEN c_acctbal >= 4000 THEN c_acctbal * 1.02
        WHEN c_acctbal >= 0 THEN c_acctbal
        ELSE 0.0
    END) AS risk_weighted_balance
FROM customer
WHERE c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
  AND c_acctbal >= 100.0 - 1500.0
GROUP BY c_mktsegment, c_nationkey
ORDER BY sum_acctbal DESC
LIMIT 100
