# ไฟล์: scripts/02_setup_workloads_and_ast.py (75-Query Factorial Benchmark: 25 Seeds x 3 Variations)
import shutil
from pathlib import Path
import pandas as pd
import sqlglot
from sqlglot import exp

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# พารามิเตอร์สำหรับ 3 Variations (V1: High Selectivity, V2: Moderate, V3: Adversarial Full-Horizon)
VARIATION_PARAMS = {
    "v1_selective": {
        "v_idx": "V1",
        "v_label": "V1: High Selectivity (Narrow)",
        "start_date": "1995-01-01",
        "end_date": "1995-03-31",
        "qty_min": 40,
        "disc_min": "0.06",
        "disc_max": "0.08",
        "seg_list": "('AUTOMOBILE')",
        "region_list": "('ASIA')",
        "acct_min": 7500.0,
        "top_limit": 25
    },
    "v2_moderate": {
        "v_idx": "V2",
        "v_label": "V2: Medium Selectivity (Moderate)",
        "start_date": "1994-01-01",
        "end_date": "1995-12-31",
        "qty_min": 24,
        "disc_min": "0.04",
        "disc_max": "0.08",
        "seg_list": "('AUTOMOBILE', 'BUILDING', 'MACHINERY')",
        "region_list": "('ASIA', 'EUROPE', 'AMERICA')",
        "acct_min": 3500.0,
        "top_limit": 50
    },
    "v3_adversarial": {
        "v_idx": "V3",
        "v_label": "V3: Low Selectivity (Full-Horizon)",
        "start_date": "1992-01-01",
        "end_date": "1998-12-01",
        "qty_min": 5,
        "disc_min": "0.01",
        "disc_max": "0.10",
        "seg_list": "('AUTOMOBILE', 'BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE')",
        "region_list": "('ASIA', 'EUROPE', 'AMERICA', 'AFRICA', 'MIDDLE EAST')",
        "acct_min": 100.0,
        "top_limit": 100
    }
}

# 25 Canonical Seed Templates (5 Tiers x 5 Seeds)
SEED_TEMPLATES = {
    "L1_scan_filter": {
        "s1_pricing_agg": """
SELECT
    l_returnflag,
    l_linestatus,
    COUNT(*) AS count_order,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1.0 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc
FROM lineitem
WHERE l_shipdate >= DATE '{start_date}'
  AND l_shipdate <= DATE '{end_date}'
  AND l_quantity >= {qty_min}
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
""",
        "s2_monotonic_case": """
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
    SUM(CASE WHEN l_discount BETWEEN {disc_min} AND {disc_max} AND l_quantity >= {qty_min} THEN l_extendedprice * l_discount ELSE 0.0 END) AS promo_discount_loss
FROM lineitem
WHERE l_shipdate >= DATE '{start_date}'
  AND l_shipdate <= DATE '{end_date}'
GROUP BY l_shipmode, l_returnflag
ORDER BY tiered_adjusted_revenue DESC
""",
        "s3_sargable_audit": """
SELECT
    l_linestatus,
    l_shipinstruct,
    COUNT(DISTINCT l_orderkey) AS distinct_orders,
    COUNT(DISTINCT l_partkey) AS distinct_parts,
    SUM(l_quantity) AS total_units,
    AVG(l_extendedprice / CASE WHEN l_quantity > 0 THEN l_quantity ELSE 1.0 END) AS avg_unit_price,
    MAX(l_extendedprice * (1.0 + l_tax)) AS max_taxed_line
FROM lineitem
WHERE l_shipdate >= DATE '{start_date}'
  AND l_shipdate <= DATE '{end_date}'
  AND l_discount >= {disc_min}
  AND l_discount <= {disc_max}
  AND l_quantity >= {qty_min}
GROUP BY l_linestatus, l_shipinstruct
HAVING COUNT(DISTINCT l_orderkey) > 5
ORDER BY distinct_orders DESC
""",
        "s4_order_priority_cube": """
SELECT
    o_orderpriority,
    o_orderstatus,
    COUNT(*) AS order_cnt,
    COUNT(DISTINCT o_custkey) AS unique_customers,
    SUM(o_totalprice) AS total_order_value,
    AVG(o_totalprice) AS mean_order_value,
    SUM(CASE WHEN o_totalprice >= 250000 THEN 1 WHEN o_totalprice >= 100000 THEN 1 ELSE 0 END) AS high_value_orders
FROM orders
WHERE o_orderdate >= DATE '{start_date}'
  AND o_orderdate <= DATE '{end_date}'
  AND o_totalprice >= {acct_min}
GROUP BY o_orderpriority, o_orderstatus
ORDER BY total_order_value DESC
""",
        "s5_customer_segment_dist": """
SELECT
    c_mktsegment,
    c_nationkey,
    COUNT(*) AS customer_cnt,
    SUM(c_acctbal) AS sum_acctbal,
    AVG(c_acctbal) AS avg_acctbal,
    MIN(c_acctbal) AS min_acctbal,
    MAX(c_acctbal) AS max_acctbal,
    SUM(CASE
        WHEN c_acctbal >= 8000 THEN c_acctbal * 1.05
        WHEN c_acctbal >= 4000 THEN c_acctbal * 1.02
        WHEN c_acctbal >= 0 THEN c_acctbal
        ELSE 0.0
    END) AS risk_weighted_balance
FROM customer
WHERE c_mktsegment IN {seg_list}
  AND c_acctbal >= {acct_min} - 1500.0
GROUP BY c_mktsegment, c_nationkey
ORDER BY sum_acctbal DESC
LIMIT {top_limit}
"""
    },

    "L2_multi_joins": {
        "s1_star_4way_revenue": """
SELECT
    c.c_mktsegment,
    n.n_name AS customer_nation,
    COUNT(DISTINCT o.o_orderkey) AS order_count,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS net_revenue,
    AVG(l.l_quantity) AS mean_quantity
FROM customer c
INNER JOIN nation n ON c.c_nationkey = n.n_nationkey
INNER JOIN orders o ON c.c_custkey = o.o_custkey
INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_mktsegment IN {seg_list}
  AND o.o_orderdate >= DATE '{start_date}'
  AND o.o_orderdate <= DATE '{end_date}'
  AND l.l_quantity >= {qty_min}
GROUP BY c.c_mktsegment, n.n_name
ORDER BY net_revenue DESC
LIMIT {top_limit}
""",
        "s2_snowflake_6way_supply": """
SELECT
    r.r_name AS region_name,
    n.n_name AS nation_name,
    p.p_mfgr,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue,
    SUM(ps.ps_supplycost * l.l_quantity) AS total_supply_cost,
    SUM((l.l_extendedprice * (1.0 - l.l_discount)) - (ps.ps_supplycost * l.l_quantity)) AS gross_margin
FROM region r
INNER JOIN nation n ON r.r_regionkey = n.n_regionkey
INNER JOIN supplier s ON n.n_nationkey = s.s_nationkey
INNER JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
INNER JOIN part p ON ps.ps_partkey = p.p_partkey
INNER JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE r.r_name IN {region_list}
  AND l.l_shipdate >= DATE '{start_date}'
  AND l.l_shipdate <= DATE '{end_date}'
  AND l.l_quantity >= {qty_min}
GROUP BY r.r_name, n.n_name, p.p_mfgr
ORDER BY gross_margin DESC
LIMIT {top_limit}
""",
        "s3_full_8table_bilateral": """
SELECT
    r_cust.r_name AS cust_region,
    n_cust.n_name AS cust_nation,
    n_supp.n_name AS supp_nation,
    p.p_type,
    COUNT(*) AS shipment_lines,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS bilateral_volume,
    SUM(ps.ps_supplycost * l.l_quantity) AS bilateral_cost
FROM customer c
INNER JOIN nation n_cust ON c.c_nationkey = n_cust.n_nationkey
INNER JOIN region r_cust ON n_cust.n_regionkey = r_cust.r_regionkey
INNER JOIN orders o ON c.c_custkey = o.o_custkey
INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
INNER JOIN supplier s ON l.l_suppkey = s.s_suppkey
INNER JOIN nation n_supp ON s.s_nationkey = n_supp.n_nationkey
INNER JOIN part p ON l.l_partkey = p.p_partkey
INNER JOIN partsupp ps ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
WHERE r_cust.r_name IN {region_list}
  AND c.c_mktsegment IN {seg_list}
  AND l.l_shipdate >= DATE '{start_date}'
  AND l.l_shipdate <= DATE '{end_date}'
  AND l.l_quantity >= {qty_min}
GROUP BY r_cust.r_name, n_cust.n_name, n_supp.n_name, p.p_type
ORDER BY bilateral_volume DESC
LIMIT {top_limit}
""",
        "s4_inequality_late_delivery": """
SELECT
    n.n_name AS supplier_nation,
    o.o_orderpriority,
    COUNT(DISTINCT o.o_orderkey) AS delayed_orders,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS delayed_revenue,
    AVG(l.l_quantity) AS avg_delayed_qty
FROM supplier s
INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
INNER JOIN region r ON n.n_regionkey = r.r_regionkey
INNER JOIN lineitem l ON s.s_suppkey = l.l_suppkey
INNER JOIN orders o ON l.l_orderkey = o.o_orderkey
INNER JOIN customer c ON o.o_custkey = c.c_custkey
WHERE r.r_name IN {region_list}
  AND c.c_mktsegment IN {seg_list}
  AND o.o_orderdate >= DATE '{start_date}'
  AND o.o_orderdate <= DATE '{end_date}'
  AND l.l_receiptdate > l.l_commitdate
  AND l.l_shipdate <= l.l_receiptdate
  AND l.l_quantity >= {qty_min}
GROUP BY n.n_name, o.o_orderpriority
ORDER BY delayed_orders DESC
LIMIT {top_limit}
""",
        "s5_composite_partsupp_margin": """
SELECT
    p.p_brand,
    p.p_container,
    n.n_name AS nation_name,
    COUNT(*) AS line_cnt,
    SUM(l.l_quantity) AS sum_qty,
    SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS net_sales,
    SUM(ps.ps_supplycost * l.l_quantity) AS supply_expense
FROM part p
INNER JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
INNER JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
INNER JOIN lineitem l ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey
WHERE l.l_shipdate >= DATE '{start_date}'
  AND l.l_shipdate <= DATE '{end_date}'
  AND l.l_discount BETWEEN {disc_min} AND {disc_max}
  AND l.l_quantity >= {qty_min}
GROUP BY p.p_brand, p.p_container, n.n_name
ORDER BY net_sales DESC
LIMIT {top_limit}
"""
    },

    "L3_nested_cte": {
        "s1_multistage_customer_tier": """
WITH stage1_filtered_lines AS (
    SELECT
        l_orderkey,
        SUM(l_extendedprice * (1.0 - l_discount)) AS order_net_rev,
        SUM(l_quantity) AS order_qty,
        COUNT(*) AS line_items_cnt
    FROM lineitem
    WHERE l_shipdate >= DATE '{start_date}'
      AND l_shipdate <= DATE '{end_date}'
      AND l_quantity >= {qty_min}
    GROUP BY l_orderkey
),
stage2_customer_rollup AS (
    SELECT
        c.c_custkey,
        c.c_mktsegment,
        c.c_nationkey,
        COUNT(o.o_orderkey) AS valid_orders,
        SUM(s1.order_net_rev) AS cust_lifetime_rev,
        AVG(s1.order_qty) AS avg_order_qty
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    INNER JOIN stage1_filtered_lines s1 ON o.o_orderkey = s1.l_orderkey
    WHERE c.c_mktsegment IN {seg_list}
    GROUP BY c.c_custkey, c.c_mktsegment, c.c_nationkey
),
stage3_nation_benchmarks AS (
    SELECT
        c_nationkey,
        AVG(cust_lifetime_rev) AS nation_avg_rev
    FROM stage2_customer_rollup
    GROUP BY c_nationkey
)
SELECT
    n.n_name,
    s2.c_mktsegment,
    COUNT(s2.c_custkey) AS elite_customer_count,
    SUM(s2.cust_lifetime_rev) AS total_elite_revenue,
    AVG(s2.cust_lifetime_rev - s3.nation_avg_rev) AS avg_surplus_over_nation
FROM stage2_customer_rollup s2
INNER JOIN stage3_nation_benchmarks s3 ON s2.c_nationkey = s3.c_nationkey
INNER JOIN nation n ON s2.c_nationkey = n.n_nationkey
WHERE s2.cust_lifetime_rev > s3.nation_avg_rev
GROUP BY n.n_name, s2.c_mktsegment
ORDER BY total_elite_revenue DESC
LIMIT {top_limit}
""",
        "s2_one_grain_per_cte_fan_guard": """
WITH cte_order_grain AS (
    SELECT
        o_custkey,
        COUNT(o_orderkey) AS total_orders,
        SUM(o_totalprice) AS sum_order_totalprice,
        MAX(o_orderdate) AS latest_order_date
    FROM orders
    WHERE o_orderdate >= DATE '{start_date}'
      AND o_orderdate <= DATE '{end_date}'
    GROUP BY o_custkey
),
cte_shipment_grain AS (
    SELECT
        o.o_custkey,
        COUNT(l.l_linenumber) AS total_shipped_lines,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS sum_line_net_revenue,
        SUM(l.l_quantity) AS sum_line_qty
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= DATE '{start_date}'
      AND l.l_shipdate <= DATE '{end_date}'
      AND l.l_quantity >= {qty_min}
    GROUP BY o.o_custkey
)
SELECT
    c.c_mktsegment,
    n.n_name AS nation_name,
    COUNT(c.c_custkey) AS active_customers,
    SUM(og.total_orders) AS cohort_orders,
    SUM(og.sum_order_totalprice) AS cohort_order_book_val,
    SUM(sg.sum_line_net_revenue) AS cohort_shipped_rev,
    SUM(sg.sum_line_qty) AS cohort_shipped_qty
FROM customer c
INNER JOIN nation n ON c.c_nationkey = n.n_nationkey
INNER JOIN cte_order_grain og ON c.c_custkey = og.o_custkey
INNER JOIN cte_shipment_grain sg ON c.c_custkey = sg.o_custkey
WHERE c.c_mktsegment IN {seg_list}
  AND c.c_acctbal >= {acct_min} - 1000.0
GROUP BY c.c_mktsegment, n.n_name
ORDER BY cohort_shipped_rev DESC
LIMIT {top_limit}
""",
        "s3_null_safe_antijoin_exclusivity": """
WITH active_shipped_suppliers AS (
    SELECT DISTINCT l_suppkey
    FROM lineitem
    WHERE l_shipdate >= DATE '{start_date}'
      AND l_shipdate <= DATE '{end_date}'
      AND l_quantity >= {qty_min}
      AND l_suppkey IS NOT NULL
),
delayed_fault_suppliers AS (
    SELECT DISTINCT l_suppkey
    FROM lineitem
    WHERE l_shipdate >= DATE '{start_date}'
      AND l_shipdate <= DATE '{end_date}'
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
WHERE r.r_name IN {region_list}
GROUP BY r.r_name, n.n_name
ORDER BY total_inventory_capital DESC
LIMIT {top_limit}
""",
        "s4_correlated_part_cost_minima": """
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
WHERE r.r_name IN {region_list}
  AND l.l_shipdate >= DATE '{start_date}'
  AND l.l_shipdate <= DATE '{end_date}'
  AND l.l_quantity >= {qty_min}
GROUP BY n.n_name, os.p_brand
ORDER BY fulfilled_revenue DESC
LIMIT {top_limit}
""",
        "s5_4stage_revenue_reconciliation": """
WITH raw_line_ledger AS (
    SELECT
        l_orderkey,
        l_suppkey,
        l_extendedprice * (1.0 - l_discount) AS net_amt,
        l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS gross_taxed_amt
    FROM lineitem
    WHERE l_shipdate >= DATE '{start_date}'
      AND l_shipdate <= DATE '{end_date}'
      AND l_quantity >= {qty_min}
),
order_level_ledger AS (
    SELECT
        o.o_orderkey,
        o.o_custkey,
        o.o_orderpriority,
        SUM(rll.net_amt) AS order_net,
        SUM(rll.gross_taxed_amt) AS order_gross,
        COUNT(DISTINCT rll.l_suppkey) AS supplier_diversity
    FROM orders o
    INNER JOIN raw_line_ledger rll ON o.o_orderkey = rll.l_orderkey
    GROUP BY o.o_orderkey, o.o_custkey, o.o_orderpriority
),
customer_priority_summary AS (
    SELECT
        c.c_nationkey,
        c.c_mktsegment,
        oll.o_orderpriority,
        COUNT(oll.o_orderkey) AS order_cnt,
        SUM(oll.order_net) AS sum_net,
        SUM(oll.order_gross) AS sum_gross,
        AVG(oll.supplier_diversity) AS mean_supp_div
    FROM customer c
    INNER JOIN order_level_ledger oll ON c.c_custkey = oll.o_custkey
    WHERE c.c_mktsegment IN {seg_list}
    GROUP BY c.c_nationkey, c.c_mktsegment, oll.o_orderpriority
),
global_priority_baseline AS (
    SELECT
        o_orderpriority,
        AVG(sum_net) AS global_avg_priority_net
    FROM customer_priority_summary
    GROUP BY o_orderpriority
)
SELECT
    n.n_name AS nation_name,
    cps.c_mktsegment,
    cps.o_orderpriority,
    cps.order_cnt,
    cps.sum_net,
    cps.sum_gross,
    cps.sum_net - gpb.global_avg_priority_net AS net_vs_global_benchmark
FROM customer_priority_summary cps
INNER JOIN global_priority_baseline gpb ON cps.o_orderpriority = gpb.o_orderpriority
INNER JOIN nation n ON cps.c_nationkey = n.n_nationkey
ORDER BY cps.sum_net DESC
LIMIT {top_limit}
"""
    },

    "L4_window_analytical": {
        "s1_multi_axis_rank_pareto": """
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
    WHERE r.r_name IN {region_list}
      AND l.l_shipdate >= DATE '{start_date}'
      AND l.l_shipdate <= DATE '{end_date}'
      AND l.l_quantity >= {qty_min}
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
LIMIT {top_limit}
""",
        "s2_sliding_frame_lead_lag": """
WITH daily_shipmode_series AS (
    SELECT
        l_shipmode,
        l_shipdate,
        COUNT(*) AS daily_lines,
        SUM(l_extendedprice * (1.0 - l_discount)) AS daily_rev
    FROM lineitem
    WHERE l_shipdate >= DATE '{start_date}'
      AND l_shipdate <= DATE '{end_date}'
      AND l_quantity >= {qty_min}
    GROUP BY l_shipmode, l_shipdate
),
temporal_window_metrics AS (
    SELECT
        l_shipmode,
        l_shipdate,
        daily_rev,
        LAG(daily_rev, 1, 0.0) OVER (PARTITION BY l_shipmode ORDER BY l_shipdate) AS prev_day_rev,
        LEAD(daily_rev, 1, 0.0) OVER (PARTITION BY l_shipmode ORDER BY l_shipdate) AS next_day_rev,
        AVG(daily_rev) OVER (
            PARTITION BY l_shipmode
            ORDER BY l_shipdate
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS rolling_5day_avg,
        SUM(daily_rev) OVER (
            PARTITION BY l_shipmode
            ORDER BY l_shipdate
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_mode_rev
    FROM daily_shipmode_series
)
SELECT
    l_shipmode,
    l_shipdate,
    daily_rev,
    prev_day_rev,
    daily_rev - prev_day_rev AS dod_rev_delta,
    rolling_5day_avg,
    cumulative_mode_rev
FROM temporal_window_metrics
ORDER BY l_shipmode, l_shipdate DESC
LIMIT {top_limit}
""",
        "s3_ntile_decile_stratification": """
WITH customer_spend_summary AS (
    SELECT
        c.c_custkey,
        c.c_mktsegment,
        c.c_nationkey,
        COUNT(o.o_orderkey) AS orders_placed,
        SUM(o.o_totalprice) AS total_spend
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE c.c_mktsegment IN {seg_list}
      AND o.o_orderdate >= DATE '{start_date}'
      AND o.o_orderdate <= DATE '{end_date}'
      AND c.c_acctbal >= {acct_min} - 1500.0
    GROUP BY c.c_custkey, c.c_mktsegment, c.c_nationkey
),
decile_partitioned AS (
    SELECT
        c_custkey,
        c_mktsegment,
        c_nationkey,
        orders_placed,
        total_spend,
        NTILE(10) OVER (PARTITION BY c_mktsegment ORDER BY total_spend DESC) AS spend_decile,
        AVG(total_spend) OVER (PARTITION BY c_mktsegment, c_nationkey) AS nat_seg_mean_spend,
        MAX(total_spend) OVER (PARTITION BY c_mktsegment) AS seg_max_spend
    FROM customer_spend_summary
)
SELECT
    c_mktsegment,
    spend_decile,
    COUNT(c_custkey) AS customer_cnt,
    SUM(total_spend) AS decile_total_spend,
    AVG(total_spend) AS decile_mean_spend,
    AVG(total_spend - nat_seg_mean_spend) AS mean_diff_from_nation_avg
FROM decile_partitioned
GROUP BY c_mktsegment, spend_decile
ORDER BY c_mktsegment, spend_decile
""",
        "s4_inter_order_cadence_deltas": """
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
    WHERE c.c_mktsegment IN {seg_list}
      AND o.o_orderdate >= DATE '{start_date}'
      AND o.o_orderdate <= DATE '{end_date}'
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
""",
        "s5_part_brand_cumulative_waterfall": """
WITH brand_part_revenue AS (
    SELECT
        p.p_mfgr,
        p.p_brand,
        p.p_partkey,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS part_rev,
        SUM(l.l_quantity) AS part_qty
    FROM part p
    INNER JOIN lineitem l ON p.p_partkey = l.l_partkey
    WHERE l.l_shipdate >= DATE '{start_date}'
      AND l.l_shipdate <= DATE '{end_date}'
      AND l.l_quantity >= {qty_min}
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
LIMIT {top_limit}
"""
    },

    "L5_array_unnest": {
        "s1_order_shipmode_array_explosion": """
WITH order_packed_arrays AS (
    SELECT
        o.o_orderpriority,
        o.o_orderkey,
        ARRAY_AGG(l.l_shipmode) AS shipmode_arr,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS order_net_val
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= DATE '{start_date}'
      AND o.o_orderdate <= DATE '{end_date}'
      AND l.l_quantity >= {qty_min}
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
LIMIT {top_limit}
""",
        "s2_supplier_brand_bag_unnest": """
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
    WHERE r.r_name IN {region_list}
      AND s.s_acctbal >= {acct_min} - 1500.0
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
LIMIT {top_limit}
""",
        "s3_customer_priority_history_unnest": """
WITH customer_priority_histories AS (
    SELECT
        c.c_mktsegment,
        c.c_custkey,
        ARRAY_AGG(o.o_orderpriority) AS priority_history_arr,
        SUM(o.o_totalprice) AS customer_period_spend,
        COUNT(o.o_orderkey) AS order_cnt
    FROM customer c
    INNER JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE c.c_mktsegment IN {seg_list}
      AND o.o_orderdate >= DATE '{start_date}'
      AND o.o_orderdate <= DATE '{end_date}'
      AND c.c_acctbal >= {acct_min} - 1000.0
    GROUP BY c.c_mktsegment, c.c_custkey
    HAVING COUNT(o.o_orderkey) >= 2
)
SELECT
    cph.c_mktsegment,
    priority_tag,
    COUNT(*) AS priority_tag_instances,
    COUNT(DISTINCT cph.c_custkey) AS unique_customers,
    AVG(cph.customer_period_spend) AS avg_customer_spend
FROM customer_priority_histories cph
CROSS JOIN UNNEST(cph.priority_history_arr) AS priority_tag
GROUP BY cph.c_mktsegment, priority_tag
ORDER BY priority_tag_instances DESC
""",
        "s4_part_returnflag_vector_unnest": """
WITH part_return_vectors AS (
    SELECT
        p.p_type,
        p.p_partkey,
        ARRAY_AGG(l.l_returnflag) AS return_flag_vec,
        SUM(l.l_quantity) AS total_part_qty,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS total_part_rev
    FROM part p
    INNER JOIN lineitem l ON p.p_partkey = l.l_partkey
    WHERE l.l_shipdate >= DATE '{start_date}'
      AND l.l_shipdate <= DATE '{end_date}'
      AND l.l_quantity >= {qty_min}
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
LIMIT {top_limit}
""",
        "s5_multistage_nested_array_reagg": """
WITH stage1_order_tokens AS (
    SELECT
        o.o_custkey,
        o.o_orderkey,
        ARRAY_AGG(l.l_shipinstruct) AS instruct_arr,
        SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS order_rev
    FROM orders o
    INNER JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= DATE '{start_date}'
      AND l.l_shipdate <= DATE '{end_date}'
      AND l.l_discount BETWEEN {disc_min} AND {disc_max}
      AND l.l_quantity >= {qty_min}
    GROUP BY o.o_custkey, o.o_orderkey
),
stage2_unnested_instructions AS (
    SELECT
        s1.o_custkey,
        inst_token,
        COUNT(*) AS token_freq,
        SUM(s1.order_rev) AS token_rev
    FROM stage1_order_tokens s1
    CROSS JOIN UNNEST(s1.instruct_arr) AS inst_token
    GROUP BY s1.o_custkey, inst_token
),
stage3_customer_token_bag AS (
    SELECT
        c.c_mktsegment,
        s2.o_custkey,
        ARRAY_AGG(s2.inst_token) AS distinct_inst_bag,
        SUM(s2.token_rev) AS total_cust_token_rev
    FROM customer c
    INNER JOIN stage2_unnested_instructions s2 ON c.c_custkey = s2.o_custkey
    WHERE c.c_mktsegment IN {seg_list}
    GROUP BY c.c_mktsegment, s2.o_custkey
)
SELECT
    s3.c_mktsegment,
    final_inst,
    COUNT(DISTINCT s3.o_custkey) AS customer_reach,
    SUM(s3.total_cust_token_rev) AS aggregate_Instruction_revenue
FROM stage3_customer_token_bag s3
CROSS JOIN UNNEST(s3.distinct_inst_bag) AS final_inst
GROUP BY s3.c_mktsegment, final_inst
ORDER BY aggregate_Instruction_revenue DESC
LIMIT {top_limit}
"""
    }
}

def compute_ast_profile(sql_text: str) -> dict:
    tree = sqlglot.parse_one(sql_text, read="postgres")

    def get_max_depth(node) -> int:
        if not isinstance(node, exp.Expression):
            return 0
        child_depths = [get_max_depth(child) for child in node.iter_expressions()]
        return 1 + (max(child_depths) if child_depths else 0)

    all_nodes = list(tree.walk())
    joins = list(tree.find_all(exp.Join))
    ctes = list(tree.find_all(exp.CTE))
    subqueries = list(tree.find_all(exp.Subquery))
    windows = list(tree.find_all(exp.Window))
    cases = list(tree.find_all(exp.Case))
    unnests = list(tree.find_all(exp.Unnest))
    aggs = list(tree.find_all(exp.AggFunc))

    # ดัชนีความซับซ้อนเชิงโครงสร้างรวม (Composite Structural Complexity Index: CSCI)
    ast_depth = get_max_depth(tree)
    csci = (
        ast_depth * 1.5
        + len(joins) * 2.5
        + (len(ctes) + len(subqueries)) * 3.0
        + len(windows) * 3.5
        + len(unnests) * 4.0
        + len(cases) * 1.5
    )

    return {
        "AST_Depth": ast_depth,
        "AST_Nodes": len(all_nodes),
        "Joins": len(joins),
        "CTEs_Subqueries": len(ctes) + len(subqueries),
        "Window_Funcs": len(windows),
        "Case_Branches": len(cases),
        "Agg_Funcs": len(aggs),
        "Has_UNNEST": 1 if (len(unnests) > 0 or "UNNEST" in sql_text.upper()) else 0,
        "CSCI_Score": round(csci, 2)
    }

def build_75_workloads(root: Path):
    wl_root = root / "workloads"
    print("=" * 100)
    print(" [STEP 1] Generating 75-Query Factorial OLAP Benchmark Suite (25 Canonical Seeds x 3 Variations)")
    print("=" * 100)

    records = []
    seed_global_idx = 0

    for tier_name, seeds_dict in SEED_TEMPLATES.items():
        tier_dir = wl_root / tier_name
        if tier_dir.exists():
            shutil.rmtree(tier_dir)
        tier_dir.mkdir(parents=True, exist_ok=True)

        for s_idx, (seed_key, sql_tmpl) in enumerate(seeds_dict.items(), start=1):
            seed_global_idx += 1
            seed_id = f"S{seed_global_idx:02d}_{tier_name[:2]}_{seed_key}"

            for var_key, var_cfg in VARIATION_PARAMS.items():
                v_code = var_cfg["v_idx"]
                fname = f"q_{tier_name[:2]}_s{s_idx}_{v_code.lower()}_{seed_key}.sql"
                fpath = tier_dir / fname

                rendered_sql = sql_tmpl.strip().format(**var_cfg)
                # ยืนยันว่าแปลงข้าม Dialect (DuckDB <-> BigQuery) ได้ไร้ข้อผิดพลาด 100%
                _ = sqlglot.transpile(rendered_sql, read="postgres", write="bigquery")[0]
                fpath.write_text(rendered_sql + "\n", encoding="utf-8")

                ast_metrics = compute_ast_profile(rendered_sql)
                records.append({
                    "Query_File": fname,
                    "Complexity_Tier": tier_name,
                    "Seed_ID": seed_id,
                    "Seed_Short": f"{tier_name[:2]}_S{s_idx}",
                    "Variation_ID": v_code,
                    "Variation_Label": var_cfg["v_label"],
                    **ast_metrics
                })

    df_ast = pd.DataFrame(records)
    out_csv = wl_root / "ast_complexity_summary.csv"
    df_ast.to_csv(out_csv, index=False)

    # สรุปค่าเฉลี่ย AST แยกตามระดับความซับซ้อน L1-L5
    summary = df_ast.groupby("Complexity_Tier", as_index=False).agg(
        Queries=("Query_File", "count"),
        Seeds=("Seed_ID", "nunique"),
        Mean_Depth=("AST_Depth", "mean"),
        Max_Depth=("AST_Depth", "max"),
        Mean_Nodes=("AST_Nodes", "mean"),
        Mean_Joins=("Joins", "mean"),
        Mean_CTEs=("CTEs_Subqueries", "mean"),
        Mean_Windows=("Window_Funcs", "mean"),
        Mean_CSCI=("CSCI_Score", "mean")
    )
    print(summary.to_string(index=False))
    print("=" * 100)
    print(f"[OK] Generated {len(df_ast)} SQL Queries across {df_ast['Seed_ID'].nunique()} Canonical Seeds!")
    print(f"[OK] Saved AST Structural Profile to: {out_csv}\n")

if __name__ == "__main__":
    build_75_workloads(PROJECT_ROOT)


