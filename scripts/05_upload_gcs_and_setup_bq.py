# ไฟล์: scripts/05_upload_gcs_and_setup_bq.py (โหมดปลอดภัย BigQuery Sandbox 100% ไม่ใช้ GCS/ไม่ใช้ Billing)
import time
import yaml
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TPCH_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]

# กฎการจัดเรียงข้อมูล (Clustering) สำหรับตารางโหมด Optimized
CLUSTERING_RULES = {
    "lineitem": ["l_shipdate", "l_returnflag", "l_linestatus"],
    "orders": ["o_orderdate", "o_orderpriority", "o_custkey"],
    "customer": ["c_mktsegment", "c_nationkey"],
    "part": ["p_brand", "p_type"],
    "supplier": ["s_nationkey"]
}

def upload_local_parquet_to_bq_sandbox(root: Path):
    cfg_path = root / "config" / "benchmark_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cred_path = root / cfg["project_settings"]["credentials_path"]
    if not cred_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์กุญแจที่: {cred_path}")

    sf = cfg["experiment_controls"]["scale_factor"]
    location = cfg["project_settings"].get("gcp_location", "asia-southeast1")

    credentials = service_account.Credentials.from_service_account_file(str(cred_path))
    project_id = credentials.project_id
    bq_client = bigquery.Client(credentials=credentials, project=project_id, location=location)

    parquet_dir = root / "data" / "raw_parquet" / f"sf{sf}_{sf}gb"
    total_mb = sum(f.stat().st_size for f in parquet_dir.glob("*.parquet")) / (1024 ** 2)

    print("=" * 85)
    print(f" [SAFE MODE] BigQuery Sandbox Direct Upload (No GCS / No Billing Required)")
    print(f" Project ID : {project_id} | Location : {location}")
    print(f" Total Data : {total_mb:.2f} MB (ใช้พื้นที่เพียง {(total_mb * 2 / 10240)*100:.1f}% ของโควตาฟรี 10 GB)")
    print("=" * 85)

    # สร้าง 2 Datasets: โหมดตารางมาตรฐาน (Unoptimized) และโหมดทำ Clustering (Optimized)
    ds_unopt = f"tpch_unopt_sf{sf}"
    ds_opt = f"tpch_opt_sf{sf}"

    for ds_name in [ds_unopt, ds_opt]:
        ds_ref = bigquery.Dataset(f"{project_id}.{ds_name}")
        ds_ref.location = location
        # ในโหมด Sandbox ตารางจะมีอายุเริ่มต้น 60 วันโดยอัตโนมัติ
        bq_client.create_dataset(ds_ref, exists_ok=True)
        print(f"  -> [DATASET READY] {project_id}.{ds_name}")

    print("\n[STEP 2] Uploading Local Parquet Files Directly to BigQuery Sandbox Tables...")
    for tbl in TPCH_TABLES:
        local_file = parquet_dir / f"{tbl}.parquet"

        # 1. โหลดเข้าตารางโหมด Unoptimized (ไม่มี Clustering)
        t_id_unopt = f"{project_id}.{ds_unopt}.{tbl}"
        cfg_unopt = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        with open(local_file, "rb") as source_file:
            job1 = bq_client.load_table_from_file(source_file, t_id_unopt, job_config=cfg_unopt)
            job1.result()

        # 2. โหลดเข้าตารางโหมด Optimized (เปิด Clustering Keys)
        t_id_opt = f"{project_id}.{ds_opt}.{tbl}"
        cfg_opt = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        if tbl in CLUSTERING_RULES:
            cfg_opt.clustering_fields = CLUSTERING_RULES[tbl]

        t0 = time.perf_counter()
        with open(local_file, "rb") as source_file:
            job2 = bq_client.load_table_from_file(source_file, t_id_opt, job_config=cfg_opt)
            job2.result()
        elapsed = time.perf_counter() - t0

        dest_tbl = bq_client.get_table(t_id_opt)
        print(f"  -> [{tbl:<8}] Uploaded {dest_tbl.num_rows:>10,} rows ({elapsed:>5.2f} s) | Clustered: {dest_tbl.clustering_fields}")

    cfg["project_settings"]["gcp_project_id"] = project_id
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print("\n[COMPLETED] อัปโหลดตารางทั้ง 8 ตารางเข้า BigQuery Sandbox สำเร็จโดยไม่เสียค่าใช้จ่าย 100%!\n")

if __name__ == "__main__":
    upload_local_parquet_to_bq_sandbox(PROJECT_ROOT)
