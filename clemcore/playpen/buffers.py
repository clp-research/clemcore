class RolloutBuffer:
    def add(self, context, response, done, info):
        print(context, f"[{','.join(response)}]", f"[{','.join(str(done))}]")
