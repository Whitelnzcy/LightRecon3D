# E16 Targeted GT Sidecar and Final B/M

```text
registration_id = E16_PREREG_targeted_GT_sidecar_and_final_BM_20260730_v1
execution_authorized = true
protocol_modification_authorized = false
canonical_request_sha256 = 030f0d428b99a314f84f1bc733046bafbf20b0d2b7e66231b8221c76b0eed667
```

The canonical request bytes are stored in the Codex attachment supplied by the
user. E16 leaves E15 blocked, targets only the 636 E14-selected views, acquires
exactly one `layout.json` and one `camera_pose.txt` beside each selected RGB by
member-level HTTP Range, keeps all final GT semantics sealed until 256 B16/M16
predictions are frozen, and then performs the single registered non-mutating
comparison.
