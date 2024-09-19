# Setting up

## Clone the repository

```
git clone https://github.com/clembench/clembench.git
```

## Install dependencies

Create a virtual environment


```
python3 -m venv venv
```

Activate the environment


```
source venv/bin/activate
```

Install the required libraries (note that ```openai``` needs to be updated to version ```1.7.0``` if you already installed the requirements before 2024-01-10)

```
pip install -r requirements.txt
```

## API Key

Create a file `key.json` in the project root and paste in your api key and base url.

```
{
 	"generic_openai_compatible": {
		"api_key":"", 
		"base_url": ""
 	},
}
```

### Providers hosting models

We strive to have available a hosted model on our infrastructure (enquire with you supervisor), but can't guarantee that it's up all the time.

There are some commercial providers offering a variety of (open-weight) models; some of these offer generous inference credit when you sign up. These can be a good choice when you're starting out.


- Together.ai - 25 $
  Models list: https://docs.together.ai/docs/inference-models
  Base API URL: https://api.together.xyz/v1
- Anyscale - 20 $
  Models list: https://console.anyscale.com/v2
  Base API URL: https://api.endpoints.anyscale.com/v1
- Groq - 20 $
  Models list: https://console.groq.com/docs/models
  Base URL: https://api.groq.com/openai/v1




# Validating your installation

Go into the project root and prepare path to run from cmdline

```
source prepare_path.sh
```

Then run the cli script to run the `taboo` game on the `high_en` experiment using the pairs of `fsc-openchat-3.5-0106` models. You can replace the game and experiment names for your own use case.

```
python3 scripts/cli.py run -g taboo -m fsc-openchat-3.5-0106 -e high_en
```

(The `-m` option tells the script which model to use; since taboo is a two player game, we need both partners to be specified here.)

This should give you an output on the terminal that contains something like the following:

```
Playing games: 100%|██████████████████████████████████| 20/20 [00:48<00:00,  2.41s/it]
```

If that is the case, output (transcripts of the games played) will have been written to `results/fsc-openchat-3.5-0106--fsc-openchat-3.5-0106/taboo` (in the main directory of the code).



For example, you can use that script to get a more readable version of the game play jsons like so:

```
python3 scripts/cli.py transcribe -g taboo
```

After running this, the `results` directory will now hold html and LaTeX views of the transcripts for each episode.

Next run the following to generate scores:

```
python3 scripts/cli.py score -g taboo
```

# Evaluation

If all you need is a table with the leaderboard results (% played, main score and clemscore for each game and model), you can run:

```
python3 evaluation/bencheval.py -p results
```

You can also skip the `-p results` where the evaluation code by default looks in that folder path. The script above creates `results.csv` and `results.html`.

Each game requires a custom analysis. The notebook ```howto_evaluate.ipynb``` shows examples of how to begin evaluating games. In particular, it generates tables with an overview of all metrics and also detailed by experiment. It also reproduces Figure 10 from the paper.
