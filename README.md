# reddit-sentiment-and-data-analysis

This is a copy of the repository where I worked with a team of 3 to complete the project.

Team Members:
1. Ashish Telukunta
2. Thomas Stanton
3. Jonathan Schild 

Actual Repository Link: https://github.com/jschild01/Final-Project-Group3

## Installations

##### Requirements

The only packages needed are contained in the 'general_requirements.txt' file.
To install, run the following:
**MUST BE IN /code FIRST**

```bash
pip3 install -r general_requirements.txt
```

The trained models should be downloaded before running any files using the following link to an accessible Google Drive. There are four models: BERT, BERT using Reddit data, RoBERTa, and ELECTRA. Models should be put in Code/content/models

```bash
https://drive.google.com/drive/folders/1mwFB0R89RYZ_qlLp675Z19VXZbMWdCuK?usp=sharing
```

---

## Running project

The training and test algorithms run from the DATS_6312_NLP directory. The main.py function conveniently starts a Streamlit serve on port 8888 that can be used to run our scripts and can be executed with the below. The server code that it calls can be found in streamlit_display.py.
**MUST BE IN /code FIRST**

```bash
python3 main.py
```

---
