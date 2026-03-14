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
cd fairness-ga-advanced
conda create -n fairness-ga python=3.10 -y
conda activate fairness-ga
```

이후 이 프로젝트 작업 시마다 **먼저** `conda activate fairness-ga` 실행 후 `python`, `pip` 사용.

#### B) venv 사용

```bash
cd fairness-ga-advanced
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

프로젝트 루트(`fairness-ga-advanced`)에서 실행:

```bash
python scripts/download_adult.py
python scripts/download_benchmark_datasets.py
```

생성 파일:
- `data/adult.csv`
- `data/compas.csv`
- `data/german_credit.csv`

### 4. 실험 실행

```bash
python experiments/run_experiment.py
```

기본 설정:
- Trial 수: `30` (`src/config.py`의 `N_TRIALS`)
- Trial 당 평가 예산: `50,000` (`POPULATION_SIZE * GENERATIONS`)
- GA와 Random Search 모두 **동일 예산**으로 비교
- 데이터셋: `adult, compas, german`
- fairness metric: individual discrimination + demographic parity + equalized odds(proxy)

빠른 점검 실행 예시:
```bash
python experiments/run_experiment.py --datasets adult --trials 3 --budget 5000
python experiments/run_experiment.py --with-sensitivity --sensitivity-trials 3
```

저장 파일:
- `experiments/results/combined_results.json`
- `experiments/results/<dataset>/trial_metrics.npz`
- `experiments/results/<dataset>/convergence.npz`
- `experiments/results/<dataset>/sensitivity.json` (옵션)

### 5. 통계 검정

```bash
python experiments/statistical_test.py
```

Wilcoxon signed-rank test로 GA > Random 가설을 검정합니다.

### 6. 시각화

```bash
python experiments/visualization.py
python experiments/visualization.py --dataset compas
```

출력: `experiments/fairness_ga_results_<dataset>.png`

---

## 요약 (Conda 예시)

```bash
cd fairness-ga-advanced
conda activate fairness-ga  
pip install -r requirements.txt
python scripts/download_adult.py
python scripts/download_benchmark_datasets.py
python experiments/run_experiment.py
python experiments/statistical_test.py
python experiments/visualization.py
```

최초 한 번만: `conda create -n fairness-ga python=3.10 -y`

conda create -n fairness-ga python=3.10 -y && conda activate fairness-ga && pip install -r requirements.txt && python experiments/run_experiment.py
