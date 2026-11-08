WITH raw_line_ledger AS (
    SELECT
        l_orderkey,
        l_suppkey,
        l_extendedprice * (1.0 - l_discount) AS net_amt,
        l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS gross_taxed_amt
    FROM lineitem
    WHERE l_shipdate >= DATE '1992-01-01'
      AND l_shipdate <= DATE '1998-12-01'
      AND l_quantity >= 5
),
order_level_ledger AS (
    SELECT
        o.o_orderkey,
        o.o_custkey,
        o.o_orderpriority,
        SUM(rll.net_amt) AS order_net,
        SUM(rll.gross_taxed_amt) AS order_gross,
        COUNT(DISTINCT rll.l_suppkey) AS supplier_diversity
    FROM orders o
    INNER JOIN raw_line_ledger rll ON o.o_orderkey = rll.l_orderkey
    GROUP BY o.o_orderkey, o.o_custkey, o.o_orderpriority
),
customer_priority_summary AS (
    SELECT
        c.c_nationkey,
        c.c_mktsegment,
        oll.o_orderpriority,
        COUNT(oll.o_orderkey) AS order_cnt,
        SUM(oll.order_net) AS sum_net,
        SUM(oll.order_gross) AS sum_gross,
        AVG(oll.supplier_diversity) AS mean_supp_div
    FROM customer c
    INNER JOIN order_level_ledger oll ON c.c_custkey = oll.o_custkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
    GROUP BY c.c_nationkey, c.c_mktsegment, oll.o_orderpriority
),
global_priority_baseline AS (
    SELECT
        o_orderpriority,
        AVG(sum_net) AS global_avg_priority_net
    FROM customer_priority_summary
    GROUP BY o_orderpriority
)
SELECT
    n.n_name AS nation_name,
    cps.c_mktsegment,
    cps.o_orderpriority,
    cps.order_cnt,
    cps.sum_net,
    cps.sum_gross,
    cps.sum_net - gpb.global_avg_priority_net AS net_vs_global_benchmark
FROM customer_priority_summary cps
INNER JOIN global_priority_baseline gpb ON cps.o_orderpriority = gpb.o_orderpriority
INNER JOIN nation n ON cps.c_nationkey = n.n_nationkey
ORDER BY cps.sum_net DESC
LIMIT 100
