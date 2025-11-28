# skills testing

## overview

contains a simple path to run the model card stats yaml path in the llm eval skill.

without the instrumentation, agent is:

```python

@fast.agent(
    name="eval_skill",
    skills=["../skills/"],
    servers=["huggingface"],
    instruction=default_instruction,
)

await eval_agent.generate(load_prompt(Path("build_olmo_yaml.md")))

```

This prompt triggers the skill and asks for a YAML file containing the model benchmarks. The output is asserted with `test_eval_assertions.py` and recorded in `runs.csv`.

![](./runs-skill-v1/pass_rate_by_model.png)

Some notes:

 - The skills repo is reset after each run in case an agent pollutes the directory
 - We don't keep this directory clean, so there is potential for interference or discovering previous run data.
 - The HF MCP Server is included in context with the `skills` bouquet - so model/dataset search, repo details, jobs and docs are made available.
 - AGENTS.md is used to inform the agent that `uv` is in use, and that huggingface_hub is installed. The `.venv` isn't reset every time either, so one run installing a heavy package will pay a time penalty - might be worth checking if that's a concern.
 - To check that, all the conversations are easily parseable/grepable. 
 - The results were regraded to take in to account specific benchmark metric assertions. That script is in the `dev` folder for reference.
