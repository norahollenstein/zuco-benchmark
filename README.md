# Reading Task Classification

Data, description and challenge can all be found on [zuco-benchmark.com](https://www.zuco-benchmark.com./)

The code we used for creating the baseline results is available [here](https://github.com/norahollenstein/zuco-benchmark/blob/main/sentence-level/benchmark.py)


## Get started

1. Install pip 
2. Create a virtual environment with venv or conda
3. Activate your environment


### Dependecies

Install the dependecies:  

```pip install -r requirements.txt```


### Data

Warning: the dataset is about 70 GB
To download the data execute  
```bash get_data.sh ``` 


### Compute the benchmark baseline results

Execute 

```cd src ```  
```python benchmark_baseline.py```
