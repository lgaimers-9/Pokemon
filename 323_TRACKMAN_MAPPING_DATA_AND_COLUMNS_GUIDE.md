# 📊 [팀원 공유용] 트랙맨 상황별 매핑 데이터셋(`trackman_situational_mapping_table.csv`) 설명서

- **파일명**: `outputs/trackman_situational_mapping_table.csv`
- **행 수**: **43,307개** (KBO 6개년 동안 발생한 모든 유니크 경기 상황 조합)
- **열 수**: **24개 컬럼** (7개 매핑 기준 키 + 17개 트랙맨 물리 통계 지표)
- **용도**: `train.csv` 및 `test.csv`와 즉시 Left-Join하여 초정밀 물리 피처를 주입할 수 있는 표준 매핑 테이블

---

## 🔑 1. 7대 매핑 기준 키 (Join Keys)

이 7개 컬럼을 기준으로 `train.csv` 또는 `test.csv`와 병합(Merge)합니다:

```python
# 파이썬 Pandas 병합 예시 코드
import pandas as pd
df_test = pd.read_csv("open/data/test.csv")
df_tkm = pd.read_csv("outputs/trackman_situational_mapping_table.csv")

join_keys = ["game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
df_merged = df_test.merge(df_tkm, on=join_keys, how="left")
```

| 컬럼명 | 한글 설명 | 데이터 예시 | 의미 및 역할 |
| :--- | :--- | :---: | :--- |
| `game_month` | 경기 월 | 3 ~ 11 | 봄/여름/가을 기온과 공기 밀도에 따른 구속·무브먼트 계절성 반영 |
| `game_dayofweek` | 경기 요일 | 0 ~ 6 (월~일) | 주중 3연전 vs 주말 경기 일정 및 투수 휴식 주기 반영 |
| `inning` | 이닝 | 1 ~ 12 | 선발투수 투구수 누적 피로도 및 경기 후반 불펜 투입 반영 |
| `top_bottom` | 초/말 | 0(초) / 1(말) | 원정팀 수비 vs 홈팀 수비 상황 |
| `balls_before` | 이전 볼 카운트 | 0 ~ 3 | 투수의 볼카운트 불리/유리 상황 |
| `strikes_before` | 이전 스트라이크 카운트 | 0 ~ 2 | 투수의 2스트라이크 유인구 구사 상황 |
| `outs_before` | 이전 아웃 카운트 | 0 ~ 2 | 주자 상황 및 이닝 종료 압박감 |

---

## 🔬 2. 트랙맨 17개 물리 통계 피처 설명 (Physical Metrics)

각 상황별로 과거 147.5만 건의 투구에서 집계된 평균(`mean`) 및 표준편차(`std`)입니다:

| 컬럼명 | 물리적 의미 | 단위 | 설명 |
| :--- | :--- | :---: | :--- |
| `tkm_rel_speed_mean` | 릴리스 평균 구속 | mph | 투수가 공을 던진 순간의 평균 구속 |
| `tkm_rel_speed_std` | 릴리스 구속 표준편차 | mph | 해당 상황에서 투수들의 구속 편차 |
| `tkm_spin_rate_mean` | 분당 평균 총 회전수 | rpm | 공의 회전력 (스핀이 높을수록 헛스윙률 증가) |
| `tkm_spin_rate_std` | 총 회전수 표준편차 | rpm | 회전수 분산 |
| `tkm_induced_vert_break_mean` | 상하 수직 무브먼트 (IVB) | inch | 중력을 제외하고 공기역학(마그누스 힘)으로 솟구치는 상하 낙차 |
| `tkm_induced_vert_break_std` | 수직 무브먼트 표준편차 | inch | 상하 무브먼트 분산 |
| `tkm_horz_break_mean` | 좌우 수평 무브먼트 (HB) | inch | 투심/슬라이더 등 좌우로 휘어지는 횡 무브먼트 |
| `tkm_horz_break_std` | 수평 무브먼트 표준편차 | inch | 횡 무브먼트 분산 |
| `tkm_extension_mean` | 평균 익스텐션 거리 | ft | 투구판에서 공을 놓는 릴리스 포인트까지 앞으로 끌고 나온 거리 |
| `tkm_extension_std` | 익스텐션 표준편차 | ft | 릴리스 포인트 전후 편차 |
| `tkm_rel_height_mean` | 릴리스 상하 높이 | ft | 공을 던지는 지점의 높이 (오버핸드 vs 스리쿼터) |
| `tkm_rel_height_std` | 릴리스 높이 표준편차 | ft | 릴리스 높이 분산 |
| `tkm_rel_side_mean` | 릴리스 좌우 위치 | ft | 투구판 중심 기준 좌우 릴리스 위치 |
| `tkm_rel_side_std` | 릴리스 좌우 위치 표준편차 | ft | 좌우 릴리스 위치 분산 |
| `tkm_zone_speed_mean` | 홈플레이트 통과 속도 | mph | 포수 미트에 도달할 때의 최종 구속 (공기저항 감속 반영) |
| `tkm_zone_speed_std` | 홈플레이트 통과 속도 편차 | mph | 종속 분산 |
| `tkm_n_pitches` | 해당 상황 총 투구 표본 수 | 개 | 과거 6개년간 해당 상황 조합에서 던져진 총 공의 개수 |

---

## 💡 3. 이 매핑 데이터의 핵심 가치

1. **규정 100% 준수**: 2025년 미래 데이터를 쓰지 않고, 순수 과거 6개년 물리 베이스라인만 집계하여 결합하므로 완벽하게 합법입니다.
2. **초정밀 매칭률 (99.97%)**: 43,307개의 촘촘한 경기 상황 그리드를 통해 `test.csv`의 거의 모든 행을 빈틈없이 1:1로 채워줍니다.
3. **세이버메트릭스 파생 피처의 원천**: 이 데이터를 기반으로 **체감 유효구속($v_{\text{eff}}$), 수직 접근각도(VAA), 스핀 효율, 3D 터널링 거리** 등의 1,030점 돌파 물리 공식이 계산됩니다.
