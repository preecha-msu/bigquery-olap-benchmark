SELECT
    l_linestatus,
    l_shipinstruct,
    COUNT(DISTINCT l_orderkey) AS distinct_orders,
    COUNT(DISTINCT l_partkey) AS distinct_parts,
    SUM(l_quantity) AS total_units,
    AVG(l_extendedprice / CASE WHEN l_quantity > 0 THEN l_quantity ELSE 1.0 END) AS avg_unit_price,
    MAX(l_extendedprice * (1.0 + l_tax)) AS max_taxed_line
FROM lineitem
WHERE l_shipdate >= DATE '1992-01-01'
  AND l_shipdate <= DATE '1998-12-01'
  AND l_discount >= 0.01
  AND l_discount <= 0.10
  AND l_quantity >= 5
GROUP BY l_linestatus, l_shipinstruct
HAVING COUNT(DISTINCT l_orderkey) > 5
ORDER BY distinct_orders DESC
