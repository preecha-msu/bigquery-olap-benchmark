SELECT
    r_cust.r_name AS cust_region,
    n_cust.n_name AS cust_nation,
    n_supp.n_name AS supp_nation,
    p.p_type,
    COUNT(*) AS shipment_lines,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS bilateral_volume,
    SUM(ps.ps_supplycost * l.l_quantity) AS bilateral_cost
FROM customer c
INNER JOIN nation n_cust ON c.c_nationkey = n_cust.n_nationkey
INNER JOIN region r_cust ON n_cust.n_regionkey = r_cust.r_regionkey
INNER JOIN orders o ON c.c_custkey = o.o_custkey
INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
INNER JOIN supplier s ON l.l_suppkey = s.s_suppkey
INNER JOIN nation n_supp ON s.s_nationkey = n_supp.n_nationkey
INNER JOIN part p ON l.l_partkey = p.p_partkey
INNER JOIN partsupp ps ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
WHERE r_cust.r_name IN ('ASIA', 'EUROPE', 'AMERICA')
  AND c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY')
  AND l.l_shipdate >= DATE '1994-01-01'
  AND l.l_shipdate <= DATE '1995-12-31'
  AND l.l_quantity >= 24
GROUP BY r_cust.r_name, n_cust.n_name, n_supp.n_name, p.p_type
ORDER BY bilateral_volume DESC
LIMIT 50
