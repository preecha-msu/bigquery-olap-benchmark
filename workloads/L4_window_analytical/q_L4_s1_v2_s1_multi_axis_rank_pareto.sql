WITH supplier_nation_perf AS (
    SELECT
        r.r_name AS region_name,
        n.n_name AS nation_name,
        s.s_suppkey,
        COUNT(l.l_orderkey) AS line_count,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS supplier_revenue,
        SUM(l.l_quantity) AS supplier_qty
    FROM region r
    INNER JOIN nation n ON r.r_regionkey = n.n_regionkey
    INNER JOIN supplier s ON n.n_nationkey = s.s_nationkey
    INNER JOIN lineitem l ON s.s_suppkey = l.l_suppkey
    WHERE r.r_name IN ('ASIA', 'EUROPE', 'AMERICA')
      AND l.l_shipdate >= DATE '1994-01-01'
      AND l.l_shipdate <= DATE '1995-12-31'
      AND l.l_quantity >= 24
    GROUP BY r.r_name, n.n_name, s.s_suppkey
),
windowed_suppliers AS (
    SELECT
        region_name,
        nation_name,
        s_suppkey,
        supplier_revenue,
        supplier_qty,
        RANK() OVER (PARTITION BY nation_name ORDER BY supplier_revenue DESC) AS rev_rank_in_nation,
        DENSE_RANK() OVER (PARTITION BY region_name ORDER BY supplier_qty DESC) AS qty_rank_in_region,
        SUM(supplier_revenue) OVER (PARTITION BY nation_name) AS nation_total_rev,
        AVG(supplier_revenue) OVER (PARTITION BY region_name) AS region_avg_rev
    FROM supplier_nation_perf
)
SELECT
    region_name,
    nation_name,
    s_suppkey,
    supplier_revenue,
    rev_rank_in_nation,
    qty_rank_in_region,
    supplier_revenue / CASE WHEN nation_total_rev > 0 THEN nation_total_rev ELSE 1.0 END AS nation_rev_share
FROM windowed_suppliers
WHERE rev_rank_in_nation <= 5
ORDER BY supplier_revenue DESC
LIMIT 50
