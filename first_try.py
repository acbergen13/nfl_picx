import nfl_data_py as nfl
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
seasons = list(range(2017, 2025))
games = nfl.import_schedules(seasons)
weekly = nfl.import_weekly_data(seasons)

# Target: home team win (1 = win, 0 = loss)
games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

# ---------------------------------------------------------
# 2. Aggregate player stats to team-week level
# ---------------------------------------------------------

weekly_team = (
    weekly.groupby(['season', 'week', 'recent_team'])
    .agg({
        'passing_yards': 'sum',
        'rushing_yards': 'sum',
        'receiving_yards': 'sum',
        'fantasy_points': 'sum'
    })
    .reset_index()
    .rename(columns={'recent_team': 'team'})
)


# Home stats
home_stats = weekly_team.rename(columns={
    'passing_yards': 'home_passing_yards',
    'rushing_yards': 'home_rushing_yards',
    'receiving_yards': 'home_receiving_yards',
    'fantasy_points': 'home_fantasy_points'
})

games = games.merge(
    home_stats,
    left_on=['season','week','home_team'],
    right_on=['season','week','team'],
    how='left'
).drop(columns=['team'])

# Away stats
away_stats = weekly_team.rename(columns={
    'passing_yards': 'away_passing_yards',
    'rushing_yards': 'away_rushing_yards',
    'receiving_yards': 'away_receiving_yards',
    'fantasy_points': 'away_fantasy_points'
})

games = games.merge(
    away_stats,
    left_on=['season','week','away_team'],
    right_on=['season','week','team'],
    how='left'
).drop(columns=['team'])

# ---------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------
# Differences between home and away stats
games['diff_passing'] = games['home_passing_yards'] - games['away_passing_yards']
games['diff_rushing'] = games['home_rushing_yards'] - games['away_rushing_yards']
games['diff_receiving'] = games['home_receiving_yards'] - games['away_receiving_yards']
games['diff_fantasy'] = games['home_fantasy_points'] - games['away_fantasy_points']

# ---------------------------------------------------------
# 4. Build model dataset
# ---------------------------------------------------------
drop_cols = [
    'game_id','season','week','home_team','away_team',
    'home_score','away_score','home_win'
]
X = games.drop(columns=[c for c in drop_cols if c in games.columns])
y = games['home_win']

# ---------------------------------------------------------
# 5. Train/test with TimeSeriesSplit + collect feature importances
# ---------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
feature_importances = pd.DataFrame(0, index=X.columns, columns=["importance_sum"])

for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    brier = brier_score_loss(y_test, preds)

    print(f"Fold {i}: Accuracy={acc:.3f}, Brier={brier:.3f}")

    # Store feature importances
    feature_importances["importance_sum"] += model.feature_importances_

# Average over folds
feature_importances["importance_mean"] = feature_importances["importance_sum"] / tscv.get_n_splits()

# Sort by importance
feature_importances = feature_importances.sort_values("importance_mean", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importances.head(10))
