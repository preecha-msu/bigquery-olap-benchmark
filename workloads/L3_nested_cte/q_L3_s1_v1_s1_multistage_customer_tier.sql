WITH stage1_filtered_lines AS (
    SELECT
        l_orderkey,
        SUM(l_extendedprice * (1.0 - l_discount)) AS order_net_rev,
        SUM(l_quantity) AS order_qty,
        COUNT(*) AS line_items_cnt
    FROM lineitem
    WHERE l_shipdate >= DATE '1995-01-01'
      AND l_shipdate <= DATE '1995-03-31'
      AND l_quantity >= 40
    GROUP BY l_orderkey
),
stage2_customer_rollup AS (
    SELECT
        c.c_custkey,
        c.c_mktsegment,
        c.c_nationkey,
        COUNT(o.o_orderkey) AS valid_orders,
        SUM(s1.order_net_rev) AS cust_lifetime_rev,
        AVG(s1.order_qty) AS avg_order_qty
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    INNER JOIN stage1_filtered_lines s1 ON o.o_orderkey = s1.l_orderkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE')
    GROUP BY c.c_custkey, c.c_mktsegment, c.c_nationkey
),
stage3_nation_benchmarks AS (
    SELECT
        c_nationkey,
        AVG(cust_lifetime_rev) AS nation_avg_rev
    FROM stage2_customer_rollup
    GROUP BY c_nationkey
)
SELECT
    n.n_name,
    s2.c_mktsegment,
    COUNT(s2.c_custkey) AS elite_customer_count,
    SUM(s2.cust_lifetime_rev) AS total_elite_revenue,
    AVG(s2.cust_lifetime_rev - s3.nation_avg_rev) AS avg_surplus_over_nation
FROM stage2_customer_rollup s2
INNER JOIN stage3_nation_benchmarks s3 ON s2.c_nationkey = s3.c_nationkey
INNER JOIN nation n ON s2.c_nationkey = n.n_nationkey
WHERE s2.cust_lifetime_rev > s3.nation_avg_rev
GROUP BY n.n_name, s2.c_mktsegment
ORDER BY total_elite_revenue DESC
LIMIT 25
