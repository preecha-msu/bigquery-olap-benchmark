SELECT
    o_orderpriority,
    o_orderstatus,
    COUNT(*) AS order_cnt,
    COUNT(DISTINCT o_custkey) AS unique_customers,
    SUM(o_totalprice) AS total_order_value,
    AVG(o_totalprice) AS mean_order_value,
    SUM(CASE WHEN o_totalprice >= 250000 THEN 1 WHEN o_totalprice >= 100000 THEN 1 ELSE 0 END) AS high_value_orders
FROM orders
WHERE o_orderdate >= DATE '1994-01-01'
  AND o_orderdate <= DATE '1995-12-31'
  AND o_totalprice >= 3500.0
GROUP BY o_orderpriority, o_orderstatus
ORDER BY total_order_value DESC
