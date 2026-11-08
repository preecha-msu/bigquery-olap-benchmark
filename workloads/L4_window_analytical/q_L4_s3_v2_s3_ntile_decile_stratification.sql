WITH customer_spend_summary AS (
    SELECT
        c.c_custkey,
        c.c_mktsegment,
        c.c_nationkey,
        COUNT(o.o_orderkey) AS orders_placed,
        SUM(o.o_totalprice) AS total_spend
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY')
      AND o.o_orderdate >= DATE '1994-01-01'
      AND o.o_orderdate <= DATE '1995-12-31'
      AND c.c_acctbal >= 3500.0 - 1500.0
    GROUP BY c.c_custkey, c.c_mktsegment, c.c_nationkey
),
decile_partitioned AS (
    SELECT
        c_custkey,
        c_mktsegment,
        c_nationkey,
        orders_placed,
        total_spend,
        NTILE(10) OVER (PARTITION BY c_mktsegment ORDER BY total_spend DESC) AS spend_decile,
        AVG(total_spend) OVER (PARTITION BY c_mktsegment, c_nationkey) AS nat_seg_mean_spend,
        MAX(total_spend) OVER (PARTITION BY c_mktsegment) AS seg_max_spend
    FROM customer_spend_summary
)
SELECT
    c_mktsegment,
    spend_decile,
    COUNT(c_custkey) AS customer_cnt,
    SUM(total_spend) AS decile_total_spend,
    AVG(total_spend) AS decile_mean_spend,
    AVG(total_spend - nat_seg_mean_spend) AS mean_diff_from_nation_avg
FROM decile_partitioned
GROUP BY c_mktsegment, spend_decile
ORDER BY c_mktsegment, spend_decile
