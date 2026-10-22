# NBA-Gravity
[Presentation](https://docs.google.com/presentation/d/1b2G7H0UWvJVoDoXr90XjtynKeWOTNSG7Pz_RHgDoVsI/edit?slide=id.h6dbaa4e3a915987e_2_0#slide=id.h6dbaa4e3a915987e_2_0)
[Overleaf report](https://www.overleaf.com/project/6ad7b6ddaf4f04cb33a470d0/share#91edaa8e4a034b227650f94d9b696ec924c65c2259d0d0a7)

## Fit the gravity model

The planned model is multiple linear regression (ordinary least squares) for
the continuous official `GRAV` score. It uses the 175 complete cases in
`data/processed/player_gravity_model_data.csv`, holds out 20% of players using
a reproducible random split (seed 440), and standardizes predictors using
training-set statistics only. It reports held-out MAE, R-squared, Spearman
rank correlation, and comparisons with a calibrated equal-weight index and
single-predictor OLS baselines. It also reports five-fold cross-validation
on the 140 training players only, recomputing standardization inside each
fold and summarizing fold metrics as mean and sample standard deviation.

Run from the repository root with the project virtual environment:

```sh
.venv/bin/python src/models/train.py
```

Use `--seed` to select a different reproducible split, `--test-size` to change
the held-out fraction, or `--data` to supply a different merged CSV.
