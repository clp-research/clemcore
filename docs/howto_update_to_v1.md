# How to update your Game for v1.0-beta

## No more `model_name`, but `Model` and `ModelSpec`

The new version of the framework is now passing a `Model` to the games instead of simply a `model_name`.
You can still access the model name via `model.model_spec.model_name` if you like. As you see
every Model has a model specification attribute that defines certain properties of the model.

This means that the type hints for the games need to be updated (if you used this). 
For example for taboo we adjust the player as follows:

```diff
+ from backends import Model

class WordGuesser(Player):

-    def __init__(self, model_name):
-       super().__init__(model_name)

+    def __init__(self, backend: Model):
+        super().__init__(backend)
```

And the same happens to the `GameMaster` which now receives the models as the `player_backends` argument:

```diff

class Taboo(DialogueGameMaster):

-    def __init__(self, experiment: Dict, player_backends: List[str]):
+    def __init__(self, experiment: Dict, player_backends: List[Model]):
```

Why is this a good thing? 
- Now the models get loaded early! Meaning that we have a structured object at hand instead of just a string.
- Now we do not need to load all backends anymore at start! The `ModelSpec` carries the `backend` name to be later loaded.
- Now you can provide a json-like string as the CLI `-m` option which carries additional information about the model. A minimal string would look like `{"model_name":"openchat", "backend":"openai_compatible"}`
- Now we have a model registry that matches your given model spec with pre-defined ones. The first unifying one will be used.