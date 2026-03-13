# Fairness GA Project

공정성 탐지를 위한 유전 알고리즘 실험 프로젝트입니다.

## 환경 요구사항

- **Python**: 3.8 이상
- **패키지**: `requirements.txt` 참고

## 환경 설정 및 실행

**중요**: 아래의 `python`, `pip` 명령은 **반드시 사용할 환경이 활성화된 상태**에서 실행해야 합니다.  
콘다를 쓰는 경우 환경 활성화 없이 `python`을 실행하면 PATH에 없거나 다른 Python이 불려와 오류가 날 수 있습니다.

### 1. 환경 준비 (둘 중 하나 선택)

#### A) Conda 사용 (Anaconda/Miniconda)

```bash
cd fairness-ga
conda create -n fairness-ga python=3.10 -y
conda activate fairness-ga
```

이후 이 프로젝트 작업 시마다 **먼저** `conda activate fairness-ga` 실행 후 `python`, `pip` 사용.

#### B) venv 사용

```bash
cd fairness-ga
python -m venv .venv
```

- **Windows (PowerShell)**  
  `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD)**  
  `.venv\Scripts\activate.bat`
- **Linux/macOS**  
  `source .venv/bin/activate`

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터셋 다운로드

프로젝트 루트(`fairness-ga`)에서 실행:

```bash
python scripts/download_adult.py
```

생성 파일: `data/adult.csv` (UCI Adult 데이터)

### 4. 실험 실행

```bash
python experiments/run_experiment.py
```

기본 설정:
- Trial 수: `30` (`src/config.py`의 `N_TRIALS`)
- Trial 당 평가 예산: `50,000` (`POPULATION_SIZE * GENERATIONS`)
- GA와 Random Search 모두 **동일 예산**으로 비교

빠른 점검 실행 예시:
```bash
python experiments/run_experiment.py --trials 3 --budget 5000
```

저장 파일:
- `ga_results.npy`, `random_results.npy` (전체 점수)
- `ga_trial_counts.npy`, `random_trial_counts.npy` (trial별 불공정 사례 수)
- `ga_trial_means.npy`, `random_trial_means.npy` (trial별 평균 점수)

### 5. 통계 검정

```bash
python experiments/statistical_test.py
```

Wilcoxon signed-rank test로 GA > Random 가설을 검정합니다.

### 6. 시각화

```bash
python experiments/visualization.py
```

출력: `experiments/fairness_ga_results.png`

---

## 요약 (Conda 예시)

```bash
cd fairness-ga
conda activate fairness-ga   # 매번 작업 전에 환경 활성화
pip install -r requirements.txt
python scripts/download_adult.py
python experiments/run_experiment.py
python experiments/statistical_test.py
python experiments/visualization.py
```

최초 한 번만: `conda create -n fairness-ga python=3.10 -y`