# Installation

## From PyPI

```bash
pip install kawow
```

## From source

```bash
git clone https://github.com/LucMiaz/kawow.git
cd kawow
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.10
- [RDKit](https://www.rdkit.org/) ≥ 2022.9 (for molecular parsing and SMARTS matching)
- NumPy ≥ 1.24
- scikit-learn ≥ 1.3

RDKit is most easily installed via conda:

```bash
conda install -c conda-forge rdkit
```

or via pip (unofficial build):

```bash
pip install rdkit
```

## Verifying the installation

```python
import kawow
print(kawow.__version__)
calc = kawow.PartitionCalculator()
print(calc.predict("c1ccccc1"))
```
