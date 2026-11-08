WITH cte_order_grain AS (
    SELECT
        o_custkey,
        COUNT(o_orderkey) AS total_orders,
        SUM(o_totalprice) AS sum_order_totalprice,
        MAX(o_orderdate) AS latest_order_date
    FROM orders
    WHERE o_orderdate >= DATE '1992-01-01'
      AND o_orderdate <= DATE '1998-12-01'
    GROUP BY o_custkey
),
cte_shipment_grain AS (
    SELECT
        o.o_custkey,
        COUNT(l.l_linenumber) AS total_shipped_lines,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS sum_line_net_revenue,
        SUM(l.l_quantity) AS sum_line_qty
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= DATE '1992-01-01'
      AND l.l_shipdate <= DATE '1998-12-01'
      AND l.l_quantity >= 5
    GROUP BY o.o_custkey
)
SELECT
    c.c_mktsegment,
    n.n_name AS nation_name,
    COUNT(c.c_custkey) AS active_customers,
    SUM(og.total_orders) AS cohort_orders,
    SUM(og.sum_order_totalprice) AS cohort_order_book_val,
    SUM(sg.sum_line_net_revenue) AS cohort_shipped_rev,
    SUM(sg.sum_line_qty) AS cohort_shipped_qty
FROM customer c
INNER JOIN nation n ON c.c_nationkey = n.n_nationkey
INNER JOIN cte_order_grain og ON c.c_custkey = og.o_custkey
INNER JOIN cte_shipment_grain sg ON c.c_custkey = sg.o_custkey
WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
  AND c.c_acctbal >= 100.0 - 1000.0
GROUP BY c.c_mktsegment, n.n_name
ORDER BY cohort_shipped_rev DESC
LIMIT 100
