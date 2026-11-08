SELECT
    r.r_name AS region_name,
    n.n_name AS nation_name,
    p.p_mfgr,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue,
    SUM(ps.ps_supplycost * l.l_quantity) AS total_supply_cost,
    SUM((l.l_extendedprice * (1.0 - l.l_discount)) - (ps.ps_supplycost * l.l_quantity)) AS gross_margin
FROM region r
INNER JOIN nation n ON r.r_regionkey = n.n_regionkey
INNER JOIN supplier s ON n.n_nationkey = s.s_nationkey
INNER JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
INNER JOIN part p ON ps.ps_partkey = p.p_partkey
INNER JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE r.r_name IN ('ASIA', 'EUROPE', 'AMERICA')
  AND l.l_shipdate >= DATE '1994-01-01'
  AND l.l_shipdate <= DATE '1995-12-31'
  AND l.l_quantity >= 24
GROUP BY r.r_name, n.n_name, p.p_mfgr
ORDER BY gross_margin DESC
LIMIT 50
