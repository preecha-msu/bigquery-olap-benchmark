WITH customer_priority_histories AS (
    SELECT
        c.c_mktsegment,
        c.c_custkey,
        ARRAY_AGG(o.o_orderpriority) AS priority_history_arr,
        SUM(o.o_totalprice) AS customer_period_spend,
        COUNT(o.o_orderkey) AS order_cnt
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
      AND o.o_orderdate >= DATE '1992-01-01'
      AND o.o_orderdate <= DATE '1998-12-01'
      AND c.c_acctbal >= 100.0 - 1000.0
    GROUP BY c.c_mktsegment, c.c_custkey
    HAVING COUNT(o.o_orderkey) >= 2
)
SELECT
    cph.c_mktsegment,
    priority_tag,
    COUNT(*) AS priority_tag_instances,
    COUNT(DISTINCT cph.c_custkey) AS unique_customers,
    AVG(cph.customer_period_spend) AS avg_customer_spend
FROM customer_priority_histories cph
CROSS JOIN UNNEST(cph.priority_history_arr) AS priority_tag
GROUP BY cph.c_mktsegment, priority_tag
ORDER BY priority_tag_instances DESC
