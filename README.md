# Tree of Thoughts (DFS)
- **Task**: Game of 24
- **Difference from source code**: 
    * LMs: GPT-3.5 trubo
    * Search Algorithm: DFS
    * Process: Filter invalid results in the middle of the process

## Setup
1. Set up OpenAI API key in /src/tot/models.py
2. Install required packages:
```
cd tree-of-thought-llm
pip install -r requirements.txt
```

## Running

To run the DFS version of ToT on the Game of 24, use the following command:

```bash
python run.py  --backend gpt-35-turbo --task game24   --task_start_index 900   --task_end_index 902   --method_generate propose   --method_evaluate value   --method_select greedy  --n_evaluate_sample 3   --n_select_sample 5 

```
