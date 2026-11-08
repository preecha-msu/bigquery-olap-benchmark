SELECT
    l_shipmode,
    l_returnflag,
    COUNT(*) AS total_lines,
    SUM(CASE
        WHEN l_extendedprice * (1.0 - l_discount) >= 75000 THEN l_extendedprice * (1.0 - l_discount)
        WHEN l_extendedprice * (1.0 - l_discount) >= 50000 THEN l_extendedprice * 0.95
        WHEN l_extendedprice * (1.0 - l_discount) >= 25000 THEN l_extendedprice * 0.90
        WHEN l_extendedprice * (1.0 - l_discount) >= 10000 THEN l_extendedprice * 0.85
        ELSE l_extendedprice * 0.80
    END) AS tiered_adjusted_revenue,
    SUM(CASE WHEN l_commitdate < l_receiptdate AND l_shipinstruct = 'DELIVER IN PERSON' THEN 1 ELSE 0 END) AS late_inperson_cnt,
    SUM(CASE WHEN l_discount BETWEEN 0.06 AND 0.08 AND l_quantity >= 40 THEN l_extendedprice * l_discount ELSE 0.0 END) AS promo_discount_loss
FROM lineitem
WHERE l_shipdate >= DATE '1995-01-01'
  AND l_shipdate <= DATE '1995-03-31'
GROUP BY l_shipmode, l_returnflag
ORDER BY tiered_adjusted_revenue DESC
