SELECT
    c.c_mktsegment,
    n.n_name AS customer_nation,
    COUNT(DISTINCT o.o_orderkey) AS order_count,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS net_revenue,
    AVG(l.l_quantity) AS mean_quantity
FROM customer c
INNER JOIN nation n ON c.c_nationkey = n.n_nationkey
INNER JOIN orders o ON c.c_custkey = o.o_custkey
INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
  AND o.o_orderdate >= DATE '1992-01-01'
  AND o.o_orderdate <= DATE '1998-12-01'
  AND l.l_quantity >= 5
GROUP BY c.c_mktsegment, n.n_name
ORDER BY net_revenue DESC
LIMIT 100
