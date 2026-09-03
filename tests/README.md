# Tests

The M1 suite protects:

- accepted release identity, outer manifest and every payload checksum;
- non-release and unsupported scientific status;
- sensor channel order and units;
- exact 80D schema hash and block order;
- delta prefix and causal population rolling behavior;
- normalizer, GRU metadata/shape, ensemble order and decision semantics;
- Research-owned layer-specific continuous tolerances and exact tensor shape/dtype;
- independent batch-1 `[1,20,80]` Host execution against the deployment golden;
- exact threshold, consecutive-count, onset and final decision parity;
- rejection of raw checksum drift and semantically changed feature order even if an attacker updates the inner manifest.

The M2 suite additionally protects:

- three-member static Float32 TFLite export and exact graph inventory;
- all three Float exports passing the Research-owned numerical contract;
- exact threshold, persistence, onset, and final-decision parity;
- Float `100% CPU` and minimum INT8+softmax `100% NPU` Vela placement;
- the boundary that INT8 parity, firmware, and board execution are not complete.

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```
