import optuna

def objective(trial):
    x = trial.suggest_float("x", -10.0, 10.0)
    return (x - 2.0) ** 2

storage_url = "sqlite:///optuna_study.db"

study = optuna.create_study(
    study_name="quick_start_study",
    storage=storage_url,
    direction="minimize",
    load_if_exists=True
)

study.optimize(objective, n_trials=10)

print("Best Parameters:", study.best_params)
print("Best Value:", study.best_value)