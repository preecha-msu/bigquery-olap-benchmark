WITH active_shipped_suppliers AS (
    SELECT DISTINCT l_suppkey
    FROM lineitem
    WHERE l_shipdate >= DATE '1992-01-01'
      AND l_shipdate <= DATE '1998-12-01'
      AND l_quantity >= 5
      AND l_suppkey IS NOT NULL
),
delayed_fault_suppliers AS (
    SELECT DISTINCT l_suppkey
    FROM lineitem
    WHERE l_shipdate >= DATE '1992-01-01'
      AND l_shipdate <= DATE '1998-12-01'
      AND l_receiptdate > l_commitdate
      AND l_returnflag = 'R'
      AND l_suppkey IS NOT NULL
),
clean_supplier_inventory AS (
    SELECT
        s.s_suppkey,
        s.s_nationkey,
        s.s_acctbal,
        COUNT(ps.ps_partkey) AS part_portfolio_size,
        SUM(ps.ps_availqty * ps.ps_supplycost) AS inventory_capital
    FROM supplier s
    INNER JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    INNER JOIN active_shipped_suppliers a ON s.s_suppkey = a.l_suppkey
    LEFT JOIN delayed_fault_suppliers d ON s.s_suppkey = d.l_suppkey
    WHERE d.l_suppkey IS NULL
    GROUP BY s.s_suppkey, s.s_nationkey, s.s_acctbal
)
SELECT
    r.r_name AS region_name,
    n.n_name AS nation_name,
    COUNT(csi.s_suppkey) AS fault_free_suppliers,
    SUM(csi.part_portfolio_size) AS total_parts_covered,
    SUM(csi.inventory_capital) AS total_inventory_capital,
    AVG(csi.s_acctbal) AS mean_supplier_acctbal
FROM clean_supplier_inventory csi
INNER JOIN nation n ON csi.s_nationkey = n.n_nationkey
INNER JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE r.r_name IN ('ASIA', 'EUROPE', 'AMERICA', 'AFRICA', 'MIDDLE EAST')
GROUP BY r.r_name, n.n_name
ORDER BY total_inventory_capital DESC
LIMIT 100
