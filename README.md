# BMI_771_project
BMI 771 project

# environment requirements
Must install:
- transformers, torch, pillow, scipy

# Memory requirements 
Rough Estimates for Memory Requirements (per model, for a batch size of 1, on a CPU):

Model Type	Model Size	Estimated RAM Requirements
Small NLP Models	(e.g., DistilBERT)	2-4 GB
Medium NLP Models	(e.g., BERT-base)	8-12 GB
Large NLP Models	(e.g., BERT-large)	16-24 GB
Vision Models (Small)	(e.g., ViT-Base/32)	8-12 GB
Vision Models (Large)	(e.g., CLIP-ViT-L/14)	24-32 GB or more

Minimum Recommended System RAM: 16 GB (assuming you'll work with medium-sized models)
Preferred System RAM for Flexibility: 32 GB or more (to handle larger models or bigger batches)

## Automatically finding a Google Cloud GPU

- [Instructions on setup](https://piazza.com/class/m0h830urei132s/post/31)
- After initial setup, just run `python /local/path/to/gpu-finder/gpu-finder.py`
- Once you have a VM, run `gcloud auth login`
- Connect: `gcloud compute ssh [INSTANCE_NAME] --zone=[ZONE]`
- Set up Github SSH key
  - Generate SSH key within VM: `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
  - Copy SSH key and put in Github account: `cat ~/.ssh/id_rsa.pub`
  - Github > Settings > SSH and GPG keys > New SSH key
- Clone the repo: `git clone git@github.com:ryan-moreno/BMI_771_homework.git`
- Download data (for HW2):
  - `pip install gdown tensorboard opencv-python`
  - `cd BMI_771_homework/assignment2_release`
  - `sh download_dataset.sh`
- Copying data from Google Cloud to local environment:
  - `gcloud compute scp --recurse instance-name:/path/to/remote/folder /path/to/local/dest --zone=your-zone`

<!-- ## SSH with VS code

- [Detailed instructions](https://piazza.com/class/m0h830urei132s/post/22)
- After launching a VM, edit the VM in the Google cloud console to add an SSH key
- Add the one stored at `cat ~/.ssh/google_compute_engine.pub` (see detailed instructions if not already set up)
- After initial setup, cmd+shift+p within VS code > open SSH configuration file > update external IP address for gcloud
- cmd+shift+p within VS code > connect to host -->

## Tmux

- ssh into google cloud instance
- within google cloud instance, to create a session: `tmux new -s <session-name>`
- later, reattach using: `tmux attach -t <session-name>`
- ctrl+b then d to detach

## Git

- [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Adding an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)