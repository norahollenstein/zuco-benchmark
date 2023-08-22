# Welcome to the ZuCo Benchmark on Reading Task Classification!


## 🧭 Starting from here, you can:

📖 Read the [manuscript](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.1028824/full).

:link: Gather more information on [zuco-benchmark.com](https://zuco-benchmark.github.io/zuco-benchmark/)

💻 Look at [our code](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/benchmark_baseline.py) for creating the baseline results

🏆 Create models or custom feature sets and participate in [our challenge at EvalAI](https://eval.ai/web/challenges/challenge-page/2125/overview)

## About ZuCo and the Reading Task Classification

<img src="neuroimage.jpg" align="right"
      width="400" >

The [Zurich Cognitive Language Processing Corpus (ZuCo 2.0)](https://osf.io/2urht/) is a dataset combining EEG and eye-tracking recordings from subjects reading natural sentences as a resource for the investigation of the human reading process in adult English native speakers.

The benchmark is a cross-subject classification to distinguish between normal reading and task-specific information searching. 


## How Can I Use This Repository?

This repository is supposed to give you a starting point to participate in our challenge.  
To run the code, follow the steps: 

## Dependencies 
Version: **Python 3.7.16**    
Using newer versions may lead to conflicts with h5py. 
Should you encounter any installation difficulties, please don't hesitate to open an issue.  

1. Install pip 
2. Create a virtual environment and activate it
3. Run ```pip install -r requirements.txt```


## Data
### Whole Dataset:
⚠️ **Warning**: the complete dataset contains about 70GB of files  
You can also download individual files from the [OSF](https://osf.io/d7frw/)  
To download the whole dataset, execute  
```bash get_data.sh ```

### Classification Features:
If you do not want to download the whole dataset, you can download the extracted features for each subject and feature set.  
To do so, download [features.zip](https://drive.google.com/file/d/1epWpDF_l_1VBk7RK9pE9RlN3XLfaMBie/view?usp=sharing), place the file under ```zuco-benchmark/src/``` and unzip it.


## Computing the Baseline Results

```cd src ```  
Run the code to produce baseline predictions with the SVM.    
```python benchmark_baseline.py```  

If you downloaded the whole dataset, the script will first extract the selected feature sets from the ```.mat``` files, which will take a while.


## Participation

You can use the code in ```benchmark_baseline.py``` as a starting point and:
- Try different models.
- Use other feature sets or combinations. See [config.py](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/config.py) for available feature sets
- Create your own feature combinations. To do that, take a look at the [feature-extraction](https://github.com/norahollenstein/zuco-benchmark/blob/06636628f08db17789d65a42836f45091affaa75/src/extract_features.py#L95C2-L95C2) and add your own feature combination there. 

To experiment with different models or feature combinations, you can use the [validation.py](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/validation.py), which tests your configuration using leave-one-out cross-validation on the training data.

## Submission
If you have ```create_submission``` enabled in the [config](https://github.com/norahollenstein/zuco-benchmark/blob/main/src/config.py), ```benchmark_baseline.py``` will automatically create a submission file in the correct format.  
For the submission format, check out the [example files](https://github.com/norahollenstein/zuco-benchmark/tree/main/src/submissions).  
Head to [EvalAI](https://eval.ai/web/challenges/challenge-page/2125/submission), fill in the required information and upload your submission.json. 
