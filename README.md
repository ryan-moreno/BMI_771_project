# BMI_771_project

## Final Report 
Our final report write up can be found in our main repo as [SemanticColorDiscriminability_Final_Report_submit.pdf)](./SemanticColorDiscriminability_Final_Report_submit.pdf)

## Experiments
To execute experiments run

```{bash}
python main.py 
    --short_model_name [SHORT_MODEL_NAME]
    --text_prompt_style [PROMPT_STYLE_INT]
    [LIST_OF_TASKS]
```

Example:

```{bash}
python main.py 
    --short_model_name Microsoft/LLM2CLIP
    --text_prompt_style 1
    run_model analyze_results
```
## Analysis
After executing experiments you can utilize visual_exploration.ipynb to explore the results 

## Models and citations

### ViT-B/32

- openai/clip-vit-base-patch32

### ViT-B/16

- openai/clip-vit-base-patch16

### ViT-L/14

- openai/clip-vit-large-patch14

### Microsoft LLM2CLIP Openai-B-16

- [microsoft/LLM2CLIP-Openai-B-16](https://huggingface.co/microsoft/LLM2CLIP-Openai-B-16)

    ```{bibtex}
    @misc{huang2024llm2clippowerfullanguagemodel,
        title={LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation},
        author={Weiquan Huang and Aoqi Wu and Yifan Yang and Xufang Luo and Yuqing Yang and Liang Hu and Qi Dai and Xiyang Dai and Dongdong Chen and Chong Luo and Lili Qiu},
        year={2024},
        eprint={2411.04997},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2411.04997},
    }
    ```

- LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned

    ```{bibtex}
    @inproceedings{
        llm2vec,
        title={{LLM2V}ec: Large Language Models Are Secretly Powerful Text Encoders},
        author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
        booktitle={First Conference on Language Modeling},
        year={2024},
        url={https://openreview.net/forum?id=IW1PR7vEBf}
    }
    ```

- Model is not intended to handle numbers in text, so text prompt style 4 was not run.

### Other models to consider

- [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)
- [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)

## Setup

### Environment requirements

Note that because we're not doing anything computationally intensive, we don't need to use MPS or CUDA. This can just be performed on the CPU.

Python packages:

- transformers, torch, pillow, scipy, llm2vec

<!-- # need to install
# pip install transformers
# pip install torch
# pip install matplotlib
# pip install pandas
# pip install scipy
# pip install numpy
# pip install pillow
# pip install llm2vec
# pip install flash-attn --no-build-isolation -->

### Memory requirements

Rough Estimates for Memory Requirements (per model, for a batch size of 1, on a CPU):

Model Type	Model Size	Estimated RAM Requirements
Small NLP Models	(e.g., DistilBERT)	2-4 GB
Medium NLP Models	(e.g., BERT-base)	8-12 GB
Large NLP Models	(e.g., BERT-large)	16-24 GB
Vision Models (Small)	(e.g., ViT-Base/32)	8-12 GB
Vision Models (Large)	(e.g., CLIP-ViT-L/14)	24-32 GB or more

Minimum Recommended System RAM: 16 GB (assuming you'll work with medium-sized models)
Preferred System RAM for Flexibility: 32 GB or more (to handle larger models or bigger batches)