# SimpleSnake
A Clembench game for evaluating LLMs on 2D text-based spatial reasoning tasks.

### Usage
SimpleSnake contains three separate game variations. To run the vanilla version, execute:

```bash
clem run -g simplesnake -m <model-to-evaluate-on>
```

To run a variation of SimpleSnake that implements obstacles, run:
```bash
clem run -g simplesnake_withobstacles -m <model-to-evaluate-on>
```

A variation of the game that focuses on up-front planning instead of incremental moves can also be run using:
```bash
clem run -g simplesnake_withplanning -m <model-to-evaluate-on>
```

Clembench supports various model backends and APIs. For a list of what models are currently supported:
```bash
clem list models
```

---

### Transcription and Evaluation
SimpleSnake games can be transcribed and scored/evaluated using
```bash 
clem transcribe -g [simplesnake|simplesnake_withplanning|simplesnake_withobstacles]
```
and 
```bash 
clem score -g [simplesnake|simplesnake_withplanning|simplesnake_withobstacles] && clem eval
``` 
The resulting files are saved under ```results/```.

