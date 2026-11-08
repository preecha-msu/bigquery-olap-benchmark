WITH part_return_vectors AS (
    SELECT
        p.p_type,
        p.p_partkey,
        ARRAY_AGG(l.l_returnflag) AS return_flag_vec,
        SUM(l.l_quantity) AS total_part_qty,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS total_part_rev
    FROM part p
    INNER JOIN lineitem l ON p.p_partkey = l.l_partkey
    WHERE l.l_shipdate >= DATE '1994-01-01'
      AND l.l_shipdate <= DATE '1995-12-31'
      AND l.l_quantity >= 24
    GROUP BY p.p_type, p.p_partkey
)
SELECT
    prv.p_type,
    ret_flag,
    COUNT(*) AS flag_occurrence_cnt,
    COUNT(DISTINCT prv.p_partkey) AS distinct_parts_affected,
    SUM(prv.total_part_rev) AS weighted_part_revenue
FROM part_return_vectors prv
CROSS JOIN UNNEST(prv.return_flag_vec) AS ret_flag
GROUP BY prv.p_type, ret_flag
ORDER BY weighted_part_revenue DESC
LIMIT 50
