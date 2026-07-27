# Bundled calibrated params (cloud deploy)

Small JSON only — copied into the Docker image because `output/` is gitignored/dockerignored.

| File | Role |
|------|------|
| `calibrated_params_{PAIR}.json` | Stage-1 MC priors / score_to_* |
| `calib_oos_summary_{PAIR}.json` | Train vs holdout Brier / hit rate |

**Refresh after overnight calibration** (from repo root):

```bash
./scripts/sync_calibrated_to_deploy.sh
git add fx_report/data/calibrated/
git commit -m "Refresh bundled calibrated params"
git push
```

Runtime search order: `output/` first (local overnight), then this directory.
