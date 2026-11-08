WITH ordered_customer_timeline AS (
    SELECT
        o.o_custkey,
        c.c_mktsegment,
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        ROW_NUMBER() OVER (PARTITION BY o.o_custkey ORDER BY o.o_orderdate, o.o_orderkey) AS order_seq_num,
        LAG(o.o_totalprice, 1, 0.0) OVER (PARTITION BY o.o_custkey ORDER BY o.o_orderdate, o.o_orderkey) AS prev_order_price,
        AVG(o.o_totalprice) OVER (
            PARTITION BY o.o_custkey
            ORDER BY o.o_orderdate, o.o_orderkey
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS trailing_3order_avg
    FROM orders o
    INNER JOIN customer c ON o.o_custkey = c.c_custkey
    WHERE c.c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')
      AND o.o_orderdate >= DATE '1992-01-01'
      AND o.o_orderdate <= DATE '1998-12-01'
)
SELECT
    c_mktsegment,
    order_seq_num,
    COUNT(*) AS transition_cnt,
    AVG(o_totalprice) AS mean_current_order,
    AVG(o_totalprice - prev_order_price) AS mean_basket_expansion,
    AVG(trailing_3order_avg) AS mean_trailing_avg
FROM ordered_customer_timeline
WHERE order_seq_num >= 2 AND order_seq_num <= 10
GROUP BY c_mktsegment, order_seq_num
ORDER BY c_mktsegment, order_seq_num
