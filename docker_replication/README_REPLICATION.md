# Independent Docker Replication Package (AST-Route V13)

Run a complete self-contained replication (auto-provisioning TPC-H SF=1 via DuckDB dbgen, verifying EX=1.000 across DuckDB, Polars, and DataFusion, and emitting a SHA-256 hardware receipt) in one command:

    cd docker_replication
    docker compose up --build

Outputs are written to ./replication_output/:
- independent_replication_results.csv
- independent_replication_receipt.json