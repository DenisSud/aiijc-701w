# ABOUT:

# TODOs:

## Prep
- [x] Setup github repo 
- [x] Create a kaggle notebook and upload the dataset, sync with github
- [ ] Benchmarks for models:
	- [ ] qwen/qwen3-8b:free
	- [ ] qwen/qwen3-1.7b:free
	- [ ] deepseek/deepseek-r1-0528-qwen3-8b:free
	- [ ] deepseek/deepseek-r1-0528:free
- Read this:
	- [ ] [[2504.21801v1.pdf|DeepSeek-Prover-V2]]
	- [ ] [[2402.03300v3.pdf|DeepSeekMath]]
	- [ ]  [[1511.04636v5.pdf|Deep Reinforcement Learning with a Natural Language Action Space]]

## Data Gathering
- [ ] Inspect and clean dataset if necessary.
- [ ] Translate existing problems (from given dataset) into all included languages.
- [ ] Read how datasets for RL (and math specifically) are usually organized and managed.
- [ ] Find and organize external datasets for potential later use.

## Training
- Read up on how RL for math is generally conducted with GRPO
- Create first simple GRPO notebook
- Split data into parts:
	- complete (all the available languages)
	- sepparate ones for each language
	- *potentially splits based on task dificulty (based on baseline model performance)*
- Training small model first
	***Make sure you have baseline evals before finetuning***
	- [ ] Fine tune a small (1.7b or 0.6b qwen model) using the RL notebook
	- [ ] Evaluate and compare with baseline
	- [ ] Tweak
	- [ ] ***Repeat***
	When satisfied with improvement move to next step:
- Train larger model using tested training recipe:
	***Make sure you have baseline evals before finetuning***
	- Train the **DeepSeek-R1-0528-Qwen3-8B** model
	- Eval
	- Tweak
	- ***Repeat***
- Construct final jupyter notebook and present
