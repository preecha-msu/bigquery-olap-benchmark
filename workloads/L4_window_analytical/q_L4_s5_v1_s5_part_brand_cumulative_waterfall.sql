WITH brand_part_revenue AS (
    SELECT
        p.p_mfgr,
        p.p_brand,
        p.p_partkey,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS part_rev,
        SUM(l.l_quantity) AS part_qty
    FROM part p
    INNER JOIN lineitem l ON p.p_partkey = l.l_partkey
    WHERE l.l_shipdate >= DATE '1995-01-01'
      AND l.l_shipdate <= DATE '1995-03-31'
      AND l.l_quantity >= 40
    GROUP BY p.p_mfgr, p.p_brand, p.p_partkey
),
waterfall_ranked AS (
    SELECT
        p_mfgr,
        p_brand,
        p_partkey,
        part_rev,
        part_qty,
        ROW_NUMBER() OVER (PARTITION BY p_brand ORDER BY part_rev DESC) AS brand_part_rk,
        SUM(part_rev) OVER (
            PARTITION BY p_brand
            ORDER BY part_rev DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_brand_rev,
        SUM(part_rev) OVER (PARTITION BY p_brand) AS total_brand_rev
    FROM brand_part_revenue
)
SELECT
    p_mfgr,
    p_brand,
    COUNT(p_partkey) AS top_tier_parts,
    SUM(part_rev) AS top_tier_revenue,
    AVG(running_brand_rev / CASE WHEN total_brand_rev > 0 THEN total_brand_rev ELSE 1.0 END) AS avg_cumulative_capture_ratio
FROM waterfall_ranked
WHERE brand_part_rk <= 15
GROUP BY p_mfgr, p_brand
ORDER BY top_tier_revenue DESC
LIMIT 25
