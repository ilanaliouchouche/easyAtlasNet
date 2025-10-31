## AtlasNet Training

### Autoencoder Training Script

Use the provided CLI to launch training on a ShapeNet-style point-cloud subset:

```bash
python training_scripts/ae.py \
  --data-root data \
  --output-dir outputs/ae \
  --batch-size 16 \
  --epochs 200 \
  --num-workers 4 \
  --seed 42
```

- `--data-root` points to the root directory containing `*/points/*.pts`.
- `--output-dir` stores TensorBoard runs and checkpoints.
- Adjust the other flags as needed; defaults match the command above.

The script logs training and validation metrics (including reconstructions) to TensorBoard and saves checkpoints under `outputs/ae/checkpoints`.
