SELECT
    n.n_name AS supplier_nation,
    o.o_orderpriority,
    COUNT(DISTINCT o.o_orderkey) AS delayed_orders,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS delayed_revenue,
    AVG(l.l_quantity) AS avg_delayed_qty
FROM supplier s
INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
INNER JOIN region r ON n.n_regionkey = r.r_regionkey
INNER JOIN lineitem l ON s.s_suppkey = l.l_suppkey
INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
INNER JOIN customer c ON o.o_custkey = c.c_custkey
WHERE r.r_name IN ('ASIA')
  AND c.c_mktsegment IN ('AUTOMOBILE')
  AND o.o_orderdate >= DATE '1995-01-01'
  AND o.o_orderdate <= DATE '1995-03-31'
  AND l.l_receiptdate > l.l_commitdate
  AND l.l_shipdate <= l.l_receiptdate
  AND l.l_quantity >= 40
GROUP BY n.n_name, o.o_orderpriority
ORDER BY delayed_orders DESC
LIMIT 25
