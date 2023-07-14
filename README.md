# Welcome to the ZuCo Benchmark on Reading Task Classification!


## 🧭 Starting from here you can:

📖 Read the [manuscript](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.1028824/full).

ℹ️ Gather more information on [zuco-benchmark.com](https://zuco-benchmark.github.io/zuco-benchmark/)

💻 Look at [our code](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/benchmark.py) for creating the baseline results

🏆 Create your own models and participate in [our challenge at EvalAI](https://eval.ai/web/challenges/challenge-page/1444/overview)

## About ZuCo and the Reading Task Classification

<img src="neuroimage.jpg" align="right"
      width="400" >

The [Zurich Cognitive Language Processing Corpus (ZuCo 2.0)](https://osf.io/2urht/) is a dataset combining EEG and eye-tracking recordings from subjects reading natural sentences as a resource for the investigation of the human reading process in adult English native speakers.

The benchmark consists of a cross-subject classification to distinguish between normal reading and task-specific information searching. 


## How Can I Use This Repository?

This repository is supposed to give you a starting point to participate in our challenge.  
To run the code, follow the steps: 

### Dependecies

1. Install pip 
2. Create a virtual environment and activate it
3. Run ```pip install -r requirements.txt```


### Data

⚠️ **Warning**: the complete dataset contains about 70GB of files  
You can also download individual files from the [OSF](https://osf.io/d7frw/)  
To download the whole dataset, execute  
```bash get_data.sh ``` 


### Computing the Baseline Results

```cd src ```  
Select feature-set and other configurations in [config.py](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/config.py).  
Run the code to produce baseline predictions with the SVM and your configurations:    
```python benchmark_baseline.py```  
Use the code as a starting point for trying different models or extracting your own features. 
