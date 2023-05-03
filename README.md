# Language-Technology-Project

## 1. Create a virtual environment
(make sure you are in the base environemnt)
```
# create a clean virtual environment
conda deactivate
conda env remove -n temp-env-py3.9
conda create -n temp-env-py3.9 python=3.9 -y
conda activate temp-env-py3.9
```

## 2. Install requirements

```
pip install --upgrade pip
pip install -r requirements.txt
```