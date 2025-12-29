# Performance Profiling Report (Summary)

## 1. Summary

Total execution time of the critical path is **~9.36 seconds**. The critical bottleneck is **`src.trainer.run_comparison`**, which consumes **99.7%** of total time. Data preprocessing is performing efficiently.

## 2. Analysis and Recommendations

The latency is caused by sequential training of multiple models within a single key function.

### Critical Functions Performance Table

| Function | Calls | Total Time | Average Time | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`src.trainer.run_comparison`** | 1 | **9.36 sec** | 9.36 sec | **CRITICAL=** |
| `src.preprocessor.transform` | 6 | 0.022 sec | 0.004 sec | Nominal |

### Main Improvement Suggestions

1. **Parallelization:** Use `joblib.Parallel` or `concurrent.futures` to train models concurrently.
   * *Expected Effect:* Reduce execution time from **~9.4 sec** to **~3.5 sec**.
2. **Asynchronous Processing:** Move `run_comparison` execution to a background process (e.g., Celery) to prevent blocking the main application thread.
3. **Algorithmic Optimization:** Verify that model hyperparameters (e.g., `n_estimators`) are optimal for the data size and development stage.
