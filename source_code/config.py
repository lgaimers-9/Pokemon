"""
config.py — DACON Aimers 9th Pitcher Control Success Prediction Project Configuration.
Self-contained, relative-path configuration for evaluation server execution.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Column Definitions
ID_COL = "row_id"
TARGET_COL = "control_success"

ID_ONLY_COLS = [
    "pitcher_id",
    "batter_id"
]

CATEGORICAL_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id"
]

DERIVED_CATEGORICAL_COLS = [
    "count_code",
    "platoon_matchup"
]

TRACKMAN_MATCH_FLAG_COL = "tkm_match"

TRACKMAN_DERIVED_COLS = [
    "tkm_rel_speed_mean",
    "tkm_rel_speed_std",
    "tkm_spin_rate_mean",
    "tkm_spin_rate_std",
    "tkm_induced_vert_break_mean",
    "tkm_induced_vert_break_std",
    "tkm_horz_break_mean",
    "tkm_horz_break_std",
    "tkm_extension_mean",
    "tkm_extension_std",
    "tkm_rel_height_mean",
    "tkm_rel_height_std",
    "tkm_rel_side_mean",
    "tkm_rel_side_std",
    "tkm_zone_speed_mean",
    "tkm_zone_speed_std",
    "tkm_n_pitches",
    TRACKMAN_MATCH_FLAG_COL,
]

RAW_NUMERICAL_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate"
]

DERIVED_NUMERICAL_COLS = [
    "is_leading",
    "is_tied",
    "score_diff_abs",
    "is_scoring_position",
    "pitcher_success_trend_1g",
    "pitcher_success_trend_3g"
]

ALL_FEATURE_COLS = CATEGORICAL_COLS + RAW_NUMERICAL_COLS

EXCLUDED_FEATURE_COLS = [
    "season",
    "game_type"
]

MODEL_FEATURE_COLS = [
    c for c in (
        CATEGORICAL_COLS
        + DERIVED_CATEGORICAL_COLS
        + [TRACKMAN_MATCH_FLAG_COL]
        + RAW_NUMERICAL_COLS
        + DERIVED_NUMERICAL_COLS
        + [c for c in TRACKMAN_DERIVED_COLS if c != TRACKMAN_MATCH_FLAG_COL]
    )
    if c not in EXCLUDED_FEATURE_COLS
]

TRACKMAN_JOIN_KEYS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before"
]
