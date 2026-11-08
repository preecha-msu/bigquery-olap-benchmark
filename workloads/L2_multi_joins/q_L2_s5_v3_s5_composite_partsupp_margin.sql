SELECT
    p.p_brand,
    p.p_container,
    n.n_name AS nation_name,
    COUNT(*) AS line_cnt,
    SUM(l.l_quantity) AS sum_qty,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS net_sales,
    SUM(ps.ps_supplycost * l.l_quantity) AS supply_expense
FROM part p
INNER JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
INNER JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
INNER JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
WHERE l.l_shipdate >= DATE '1992-01-01'
  AND l.l_shipdate <= DATE '1998-12-01'
  AND l.l_discount BETWEEN 0.01 AND 0.10
  AND l.l_quantity >= 5
GROUP BY p.p_brand, p.p_container, n.n_name
ORDER BY net_sales DESC
LIMIT 100
