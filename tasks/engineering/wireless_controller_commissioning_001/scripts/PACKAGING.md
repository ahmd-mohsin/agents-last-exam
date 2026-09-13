# Packaging & task-data wiring — wireless_controller_commissioning_001

The task data (train/dev/envelope archive + hidden acceptance packs) is too large
to check into git, so it is staged via ALE's `task_data_source` mechanism. This
doc is turnkey: run the staging script, upload to your substrate, set one line in
the environment yaml.

## 1. Build the staged tree

On a host with the constructed archive (`~/wcc_archive`) and the reference build
output (`out/` from `gen_reference.py` + `sionna_link.py`):

```
python3 scripts/stage_task_data.py \
    --archive ~/wcc_archive --build-out out --radio radio_system.json \
    --out ~/wcc_task_data --password "$ALE_REFERENCE_ARCHIVE_PASSWORD"
```

Produces the canonical layout:

```
~/wcc_task_data/engineering/wireless_controller_commissioning_001/base/
├── input/                      # agent-visible
│   ├── radio_system.json
│   ├── task_brief.md
│   ├── archive/                # zarr: train, dev, envelope  (~2.2 GB)
│   ├── public_eval/            # dev_eval.py + Sionna coded_link_curve.npz
│   └── starter/                # controller.py interface stub (runnable baseline)
├── reference/                  # PLAIN  -> used by  local:  (docker), hidden by timing
│   ├── hidden/                 # 12 acceptance packs (~558 MB)
│   ├── reference_metrics.json  # contract + verified expert/naive results
│   ├── reference_controller.py # self-contained expert (coverage 1.0)
│   └── second_expert.py        # independent reconstruction (coverage 1.0)
└── reference.7z                # ENCRYPTED -> used by baked_in_sandbox / gs:// / s3://
```

## 2. Upload / bake (pick your substrate)

- Local Docker (`local:`): no upload; the host dir is read directly.
- GCS (`gs://`): `gsutil -m rsync -r ~/wcc_task_data gs://<bucket>/wcc`
- S3 (`s3://`): `aws s3 sync ~/wcc_task_data s3://<bucket>/wcc`
  (needs `s3:PutObject` on the bucket; the AwsProvider VMs read it via their
  instance profile at run time.)
- Baked image: copy `~/wcc_task_data/*` under the image's `<task_data_root>` so
  each `<domain>/<task>/<variant>/` has `input/` + `reference.7z`.

## 3. Wire the environment yaml (one line)

`task_data_source` lives in `configs/environments/<env>.yaml`, not the task card.

CURRENT UPLOAD (staged 2026-09-10; 188 objects, ~3.97 GB, us-east-2):

```
task_data_source: s3://greenland-intern-artifacts-703671891219-us-east-2-an/wcc
```

Objects verified at `wcc/engineering/wireless_controller_commissioning_001/base/`:
`input/{archive,public_eval,starter,radio_system.json,task_brief.md}` and
`reference/{hidden/pack_00..11, reference_metrics.json, reference_controller.py,
second_expert.py}`.

Other substrates:

```
# docker (plain reference/):        task_data_source: local:/abs/path/to/wcc_task_data
# gcloud (encrypted reference.7z):  task_data_source: gs://<bucket>/wcc   (+ gcs_sa_key)
# baked image:                      task_data_source: baked_in_sandbox
```

Notes for the S3 path (`ale_run/environments/task_data/s3bucket.py`):
- The S3 stager syncs a PLAIN `reference/` at eval time (hidden by TIMING, like
  GCS) — `reference.7z` is only for `baked_in_sandbox`. Both are staged so either
  works.
- Every read carries `--request-payer requester`; enable requester-pays on the
  bucket (or drop the flag if not needed).
- SUBMISSION CAVEAT: this bucket is in account 703671891219. ALE's eval VMs run
  in a different account, so either grant them cross-account read (bucket policy)
  or re-host the tree in ALE's bucket. For your own runs on the AWS provider (same
  account / instance profile with read), it works as-is.

## 4. Verify the round trip

```
# encrypted path (baked/bucket): decrypt then score the reference
7z x -p"$ALE_REFERENCE_ARCHIVE_PASSWORD" -o/tmp/ref .../base/reference.7z
python3 scripts/verify_submission.py \
    --submission scripts/reference_controller.py \
    --input-dir .../base/input --reference-dir /tmp/ref
# expect: {"normalized_score": 1.0, "passed": true}
```

The reference expert and the independent second expert both score coverage 1.0;
the starter baseline and all screened frontier models score 0.0
(see `difficulty_screen.json`).
