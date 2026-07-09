# Benchmarks

Benchmarks should emit machine-readable JSON with both numerical outputs and
timing information. CI can archive the JSON to build a small performance history
without depending on a separate dashboard service.

Run the basic benchmark locally with:

```bash
python benchmarks/basic_regressions.py --output benchmark-results.json
```

Run the low-rank hypertoroidal Fourier benchmark with:

```bash
python benchmarks/low_rank_hypertoroidal_fourier.py \
  --axis-size 3 \
  --rank 2 \
  --iterations 20 \
  --output low-rank-fourier-benchmark.json
```

The low-rank Fourier benchmark reports storage scaling, TT ranks, normalization,
and timings for identity prediction, identity update, PDF evaluation, and shifts.
