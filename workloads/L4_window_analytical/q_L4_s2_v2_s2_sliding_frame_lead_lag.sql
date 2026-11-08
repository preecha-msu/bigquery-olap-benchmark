WITH daily_shipmode_series AS (
    SELECT
        l_shipmode,
        l_shipdate,
        COUNT(*) AS daily_lines,
        SUM(l_extendedprice * (1.0 - l_discount)) AS daily_rev
    FROM lineitem
    WHERE l_shipdate >= DATE '1994-01-01'
      AND l_shipdate <= DATE '1995-12-31'
      AND l_quantity >= 24
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
LIMIT 50
