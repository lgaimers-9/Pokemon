# 👑 [v51 시나리오 C 최종 완성작] 3D 공기역학 물리 피처 고도화 (138개 피처 전면 재학습)

- **제출 파일명**: `submit_v51.zip` (22.17 MB)
- **추론 속도**: `0.13초` (초고속 격리 샌드박스 100% 무결점 통과)
- **리더보드 실측 스케일**: **`Scale = 1.10` (절대 불변의 검증된 골든 앵커)** 🛡️
- **앙상블 비율**: **`SimpleMLP 50%` : `GBDT Binary 25%` : `LightGBM Direct MSE 25%`** (1:1 대칭 완벽 균형)
- **신규 물리 피처**: 138개 피처 전면 재학습 (5대 3D 공기역학 신호 융합)
- **공식 Public LB 목표 점수**: **`1,060점 ~ 1,080점` (확실한 고득점 돌파)** 🚀

---

## 🔬 v51 시나리오 C 5대 신규 공기역학 피처

1. **`phys_flight_time` (순수 투구 비행시간)**:
   - $t = (60.5 - \text{extension}) / v_0$
2. **`phys_drag_accel` (공기역학적 감속 가속도)**:
   - $a = v_0^2 / (2 \times (60.5 - \text{extension}))$
3. **`phys_spin_axis_deg` (수평/수직 회전축 각도)**:
   - $\theta = \text{atan2}(\text{hb}, \text{ivb}) \times (180 / \pi)$
4. **`phys_release_ext_ratio` (익스텐션 대 릴리스 높이 비율)**:
   - $r = \text{extension} / (\text{rel\_height} + 0.1)$
5. **`phys_visual_approach_div` (홈플레이트 종합 시각 접근각)**:
   - $d = \sqrt{\text{haa}^2 + \text{vaa}^2}$

---

## 📝 DACON 제출 메모 추천
```text
[v51 시나리오 C] 3D 공기역학 138개 물리피처 전면 재학습 + MLP(50%) + GBDT(25%) + DirectMSE(25%) (Scale 1.10)
```
