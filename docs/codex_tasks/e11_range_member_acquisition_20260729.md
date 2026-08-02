# E11 Range Member Acquisition

Registration:

```text
E11_PREREG_RANGE_MEMBER_ACQUISITION_20260729_v1
```

Execution authorization:

```text
execution_authorized = true
```

The authoritative frozen protocol is:

```text
results/range_member_acquisition_e11_20260729_v1/registration.md
```

E11 must acquire only the frozen 128 E9 candidates by member-level HTTP
Range reads. It must not download any complete perspective archive. The small
annotation archive may be downloaded in full and must remain semantically
sealed. No model, preparation, training, fitting, matcher, or evaluation call
is authorized.
