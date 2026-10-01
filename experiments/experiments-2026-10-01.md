# Experiments — 2026-10-01

Times are local America/New_York; durations in HH:MM.

| Time | Experiment | Motivation | What Tried / Evidence | Correctness | Timing | Merge decision |
| --- | --- | --- | --- | --- | --- | --- |
| 00:15 | Split GPU convergence study across both GPUs | A single worker left GPU 1 idle and GPU 0 underutilized; independent reconstructions can share both GPUs without changing optimizer mathematics or the scientific protocol. | Added four persistent workers, two per GPU, with ten-update chunks and one CPU thread each. Stopped only the serial study worker after its checkpoint, preserving completed tuning histories. Held-out and rank jobs share the queue after tuning; six diagnostic points also run concurrently. Per-run files isolate worker writes; one coordinator merges results under an ownership lock. | Nine existing tests pass; a new concurrent-shard test reproduces serial Adam sources and moments exactly after resuming two distinct checkpoints. Initial scaling, source caps, projection, and fresh gradients retain runtime checks. GPU migration and live-worker checks pending. | No wall cutoff; retain 200-step tuning and 500–2,000-step reconstruction protocol. | Keep isolated parallel runner; no production algorithm changes. |
