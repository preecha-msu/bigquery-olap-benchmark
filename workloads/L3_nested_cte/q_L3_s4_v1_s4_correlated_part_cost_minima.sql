WITH part_supplier_matrix AS (
    SELECT
        p.p_partkey,
        p.p_brand,
        p.p_type,
        ps.ps_suppkey,
        ps.ps_supplycost,
        ps.ps_availqty,
        s.s_nationkey
    FROM part p
    INNER JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
    INNER JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    WHERE p.p_size >= 5
),
part_min_cost_benchmark AS (
    SELECT
        p_partkey,
        MIN(ps_supplycost) AS min_supplycost,
        AVG(ps_supplycost) AS avg_supplycost
    FROM part_supplier_matrix
    GROUP BY p_partkey
),
optimal_suppliers AS (
    SELECT
        psm.p_partkey,
        psm.p_brand,
        psm.ps_suppkey,
        psm.s_nationkey,
        psm.ps_supplycost,
        pmb.avg_supplycost
    FROM part_supplier_matrix psm
    INNER JOIN part_min_cost_benchmark pmb
        ON psm.p_partkey = pmb.p_partkey
       AND psm.ps_supplycost = pmb.min_supplycost
)
SELECT
    n.n_name AS supplier_nation,
    os.p_brand,
    COUNT(DISTINCT os.p_partkey) AS lowest_cost_parts_won,
    SUM(l.l_quantity) AS fulfilled_quantity,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS fulfilled_revenue,
    AVG(os.avg_supplycost - os.ps_supplycost) AS mean_cost_saving_per_unit
FROM optimal_suppliers os
INNER JOIN nation n ON os.s_nationkey = n.n_nationkey
INNER JOIN region r ON n.n_regionkey = r.r_regionkey
INNER JOIN lineitem l ON os.p_partkey = l.l_partkey AND os.ps_suppkey = l.l_suppkey
WHERE r.r_name IN ('ASIA')
  AND l.l_shipdate >= DATE '1995-01-01'
  AND l.l_shipdate <= DATE '1995-03-31'
  AND l.l_quantity >= 40
GROUP BY n.n_name, os.p_brand
ORDER BY fulfilled_revenue DESC
LIMIT 25
