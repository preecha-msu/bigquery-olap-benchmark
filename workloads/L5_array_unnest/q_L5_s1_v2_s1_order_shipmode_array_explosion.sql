WITH order_packed_arrays AS (
    SELECT
        o.o_orderpriority,
        o.o_orderkey,
        ARRAY_AGG(l.l_shipmode) AS shipmode_arr,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS order_net_val
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= DATE '1994-01-01'
      AND o.o_orderdate <= DATE '1995-12-31'
      AND l.l_quantity >= 24
    GROUP BY o.o_orderpriority, o.o_orderkey
)
SELECT
    opa.o_orderpriority,
    unnested_mode,
    COUNT(*) AS unnested_line_occurrences,
    SUM(opa.order_net_val) AS associated_order_value,
    AVG(opa.order_net_val) AS mean_order_value
FROM order_packed_arrays opa
CROSS JOIN UNNEST(opa.shipmode_arr) AS unnested_mode
GROUP BY opa.o_orderpriority, unnested_mode
ORDER BY associated_order_value DESC
LIMIT 50
