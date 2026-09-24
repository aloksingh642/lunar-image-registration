# Synthetic demo pairs

These files are procedural crater fields. They are not Chandrayaan-2, LRO, or SELENE images.

Generate them with:

```bash
python main.py generate-demo
```

Each folder contains `synthetic_source.png`, `synthetic_reference.png`, sidecars, and `ground_truth.json`. The ground-truth matrix maps source pixels to reference pixels for the simulator only.
