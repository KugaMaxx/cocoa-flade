# Hyper Real-Time Flame Detection: Dynamic Insights from Event Cameras and FlaDE Dataset

FlaDE (Flame Detection Dataset based on Event Cameras) is a dataset designed for
 event-based flame detection. Traditional RGB cameras often struggle with issues 
 such as static backgrounds, overexposure, and redundant data. Event cameras, on 
 the other hand, provide a bio-inspired alternative that conquers the aforementioned
 challenges, making them particularly well-suited for tasks like flame detection.

<span id="animation"></span>
![animation](https://raw.githubusercontent.com/KugaMaxx/cocoa-flade/master/assets/images/demonstration.gif "animation")

This repository not only provides interfaces to help read event data and annotation
 files, but also includes detection examples, evaluation tools and visualization 
 functions. For more details, please refer to our [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=wSfBvMQAAAAJ&citation_for_view=wSfBvMQAAAAJ:UeHWp8X0CEIC).


## Preliminaries

Step 1: Install dependencies.

```bash
# ensure you have sudo privileges
bash setup.sh
```

Step 2: Create a [conda](https://docs.anaconda.com/miniconda/) environment.

```bash
# minimum supported Python version is 3.8
conda create -n cocoa python=3.8
```

Step 3: Install the necessary packages.

```bash
# activate the virtual environment
conda activate cocoa

# install required dependencies
pip install -r requirements.txt

# install the dv-toolkit
pip install external/dv-toolkit/.

# install repo as cocoa_flade
pip install cocoa_flade/.
```


## How to use FlaDE

To read FlaDE data, we recommend using the following Python script:
 
```python
# this helps you read FlaDE more easily
import cocoa_flade as cocoa

# this will trigger dataset download 
# if 'FlaDE' is not found in the file_path
dataset = cocoa.FlaDE('<file_path>')

# retrieve categories
# 'id', 'name', 'colors'
cats = dataset.get_cats(key='name', query=None)  # returns all categories

# retrieve annotations
# 'id', 'name', 'scene', 'frame', 'boxes', 'partition' 
tags = dataset.get_tags(key='partition', query=['train', 'val'])  # returns 'train' and 'val'
```

**Some notes：**
- If you encounter issues with the automatic download, please visit this [link](https://drive.google.com/file/d/1rLWpY98RdBYUQ7XnbdBPqWrrBf-GTeQ5/view?usp=drive_link) and extract the files into the `<file_path>` folder.
- To use the COCO format of FlaDE, please check `<file_path>/FlaDE/flade_coco` after downloading.
- For instructions on using evaluation and visualization tools, please refer to this [tutorial](./cocoa_flade/README.md).


## Run a demo

**BEC-SVM** is an example detection model used for hyper real-time flame detection
 based on event cameras. Below is how you can run the demo:

```bash
# activate virtual environment
conda activate cocoa

# navigate to the BEC-SVM sample directory
cd ./samples/bec_svm

# set up the demo
bash setup.sh
```

Then run below code to train a svm and print detection results.

```bash
# run bec-svm demo
python3 samples/bec_svm/demo.py
```

## BibTeX

For people who use , please use below citation.

```bibtex
@article{ding2024hyper,
  title={Hyper real-time flame detection: Dynamic insights from event cameras and FlaDE dataset},
  author={Ding, Saizhe and Zhang, Haorui and Zhang, Yuxin and Huang, Xinyan and Song, Weiguo},
  journal={Expert Systems with Applications},
  volume = {263},
  pages={125764},
  year={2024},
  publisher={Elsevier},
}
```

## Acknowledgments

We would like to thank [Xiang Xin](xinxiangscholar@163.com) for his valuable insights and support in this project.
