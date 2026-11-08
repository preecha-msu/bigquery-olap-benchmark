SELECT
    l_returnflag,
    l_linestatus,
    COUNT(*) AS count_order,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1.0 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc
FROM lineitem
WHERE l_shipdate >= DATE '1992-01-01'
  AND l_shipdate <= DATE '1998-12-01'
  AND l_quantity >= 5
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
