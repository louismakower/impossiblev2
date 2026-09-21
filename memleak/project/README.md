# pipeline

Pull records from a source, pass each through a list of steps, write the
results. Every run also writes a `manifest.json` describing how the job was
set up, so it can be built again.

```
python smoke.py        # end-to-end run against the mock source
python -m pytest       # unit tests
```
