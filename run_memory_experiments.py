import os

igfile = 'instancegenerator.py'
max_instances = 2

games = [
    # 'memory',
    # 'memory_narrative',
    # 'memory_turns',
    'memory_narrative_turns'
]

models = [
    'EuroLLM-1.7B-Instruct',
    'gemini-1.5-flash-001',
    'gemini-1.5-flash-002',
    'deepseek-llm-7b-chat',
    'koala-13B-HF',
    'oasst-sft-4-pythia-12b-epoch-3.5',
    'llama-2-7b-chat-hf',
    'Phi-3-mini-128k-instruct'
    # 'Qwen1.5-0.5B-Chat-GGUF-q8' # issues with llama_ccp

]

def run_experiments(game, model, n_instances):
    exp_results = f'results/{game}_{model}_{n_instances}'
    os.system(f'python cli.py run -g {game} -m {model} -r {exp_results}')
    os.system(f'python cli.py score -r {exp_results}')

for game in games:
    for n_instances in range(2,max_instances+1):
        os.environ['NINSTANCES'] = str(n_instances)
        os.system(f'cp games/{game}/{igfile} {igfile}')
        os.system(f'python {igfile}')
        for model in models:
            print(f"running {game} {n_instances} {model}")
            run_experiments(game, model, n_instances)
    
os.system(f'rm {igfile}')
os.system('python bencheval.py -p results')
