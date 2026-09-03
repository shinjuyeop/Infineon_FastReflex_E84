# Tests

The M1 suite protects:

- accepted release identity, outer manifest and every payload checksum;
- non-release and unsupported scientific status;
- sensor channel order and units;
- exact 80D schema hash and block order;
- delta prefix and causal population rolling behavior;
- normalizer, GRU metadata/shape, ensemble order and decision semantics;
- all numeric golden layers at `atol=rtol=1e-6`;
- exact threshold, consecutive-count, onset and final decision parity;
- rejection of raw checksum drift and semantically changed feature order even if an attacker updates the inner manifest.

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```
