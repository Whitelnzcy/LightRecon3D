# E16 Targeted GT Sidecar and Final B/M

```text
registration_id = E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1
execution_authorized = true
protocol_modification_authorized = false
canonical_request_sha256 = 030f0d428b99a314f84f1bc733046bafbf20b0d2b7e66231b8221c76b0eed667
```

E16 preserves the frozen E14 set of 128 scenes and 636 selected views. It
acquires exactly one `layout.json` and one `camera_pose.txt` beside each
selected RGB by member-level HTTP Range, keeps those files and the scene
annotations semantically sealed until 256 B16/M16 predictions are frozen, and
then performs the registered non-mutating final comparison.

The user-supplied attachment whose SHA-256 is recorded above is the canonical
protocol. This file is an execution index and does not replace or modify those
canonical bytes.
