# isometries

## Installation

```bash
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Dataset

```bash
sbatch dataset/dataset.sh
```

### Experiments

```bash
sbatch experiments/$DATASET_NAME/$DEFORMATION_TYPE/run.sh
```

## Licence

Based on [Neural Isometries](https://github.com/vsitzmann/neural-isometries/tree/main) and issued under the same [license](https://github.com/vsitzmann/neural-isometries/blob/main/LICENSE.txt).