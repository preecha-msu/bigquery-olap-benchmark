WITH supplier_brand_bags AS (
    SELECT
        n.n_name AS nation_name,
        s.s_suppkey,
        ARRAY_AGG(p.p_brand) AS brand_bag,
        SUM(ps.ps_availqty * ps.ps_supplycost) AS supplier_stock_val
    FROM supplier s
    INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
    INNER JOIN region r ON n.n_regionkey = r.r_regionkey
    INNER JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    INNER JOIN part p ON ps.ps_partkey = p.p_partkey
    WHERE r.r_name IN ('ASIA', 'EUROPE', 'AMERICA')
      AND s.s_acctbal >= 3500.0 - 1500.0
    GROUP BY n.n_name, s.s_suppkey
)
SELECT
    sbb.nation_name,
    unnested_brand,
    COUNT(DISTINCT sbb.s_suppkey) AS distinct_suppliers,
    COUNT(*) AS total_part_brand_slots,
    AVG(sbb.supplier_stock_val) AS mean_backing_stock_val
FROM supplier_brand_bags sbb
CROSS JOIN UNNEST(sbb.brand_bag) AS unnested_brand
GROUP BY sbb.nation_name, unnested_brand
ORDER BY total_part_brand_slots DESC
LIMIT 50
