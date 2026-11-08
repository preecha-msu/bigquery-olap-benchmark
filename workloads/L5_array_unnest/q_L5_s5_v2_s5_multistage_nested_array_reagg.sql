WITH stage1_order_tokens AS (
    SELECT
        o.o_custkey,
        o.o_orderkey,
        ARRAY_AGG(l.l_shipinstruct) AS instruct_arr,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS order_rev
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= DATE '1994-01-01'
      AND l.l_shipdate <= DATE '1995-12-31'
      AND l.l_discount BETWEEN 0.04 AND 0.08
      AND l.l_quantity >= 24
    GROUP BY o.o_custkey, o.o_orderkey
),
stage2_unnested_instructions AS (
    SELECT
        s1.o_custkey,
        inst_token,
        COUNT(*) AS token_freq,
        SUM(s1.order_rev) AS token_rev
    FROM stage1_order_tokens s1
    CROSS JOIN UNNEST(s1.instruct_arr) AS inst_token
    GROUP BY s1.o_custkey, inst_token
),
stage3_customer_token_bag AS (
    SELECT
        c.c_mktsegment,
        s2.o_custkey,
        ARRAY_AGG(s2.inst_token) AS distinct_inst_bag,
        SUM(s2.token_rev) AS total_cust_token_rev
    FROM customer c
    INNER JOIN stage2_unnested_instructions s2 ON c.c_custkey = s2.o_custkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY')
    GROUP BY c.c_mktsegment, s2.o_custkey
)
SELECT
    s3.c_mktsegment,
    final_inst,
    COUNT(DISTINCT s3.o_custkey) AS customer_reach,
    SUM(s3.total_cust_token_rev) AS aggregate_Instruction_revenue
FROM stage3_customer_token_bag s3
CROSS JOIN UNNEST(s3.distinct_inst_bag) AS final_inst
GROUP BY s3.c_mktsegment, final_inst
ORDER BY aggregate_Instruction_revenue DESC
LIMIT 50
