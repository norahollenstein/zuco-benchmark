# Reading Task Classification
<p align="left">
<img src="neuroimage.jpg" width="500" /> 
</p>

More information can be found on [zuco-benchmark.com](https://www.zuco-benchmark.com./)

The code we used for creating the baseline results is available [here](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/benchmark.py)

## Get started

1. Install pip 
2. Create a virtual environment with venv or conda
3. Activate your environment


### Dependecies

Install the dependecies:  

```pip install -r requirements.txt```


### Data

Warning: the dataset is about 70 GB  
You can also just download individual files from the [OSF](https://osf.io/d7frw/)  
To download the whole dataset, execute  
```bash get_data.sh ``` 


### Compute baseline results

```cd src ```  
Select feature-set and other configurations in [config.py](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/config.py).  
Run the code to produce baseline predictions with the SVM and your configurations:    
```python benchmark_baseline.py```  
Use the code as a starting point for trying different models or extracting your own features. 
