# E8 official archive acquisition preregistration

Registration ID:

```text
E8_PREREG_official_archive_acquisition_20260728_v1
```

Registration state:

```text
PRE_REGISTERED_NOT_AUTHORIZED
```

The authoritative full protocol is preserved byte-for-byte as
`results/official_archive_acquisition_e8_20260728_v1/registration.md`.
Its source attachment SHA-256 is:

```text
565324e8ee8ddeb5c527fe463fb5e632b6b0b242a206d431e3e01f70df561799
```

This turn freezes rules only. It does not authorize inventory execution,
network/HEAD/Range probes, ZIP central-directory reads, directory creation on
D/F, archive downloads, extraction, input preparation, model execution, GT
semantic parsing, training, or evaluation.

The registered future execution must:

* use `scene_(\d{5})(?!\d)` everywhere;
* freshly inventory E, relevant D/F Structured3D roots, manifests, outputs,
  archives, and transfer bundles before download;
* retrieve and freeze the official README and errata;
* regenerate candidates independently of E7 with the registered E7 salt;
* validate scene/archive membership from remote ZIP central directories;
* freeze all HTTP identities before download;
* satisfy D/F/E disk thresholds;
* retain all completed official archives;
* extract only the frozen 128 scenes into separated RGB and sealed-annotation
  roots;
* freeze, but not run, future DUSt3R/Stage1 input preparation;
* keep every registered forbidden model/evaluation call at zero.

A separate explicit user authorization is required before any execution.
