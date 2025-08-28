# MassGen Voting Mechanism Comparison Guide

This guide provides detailed instructions on how to use MassGen's anonymous and identified voting features for experimental comparison.

## 🎭 Voting Mechanism Overview

MassGen supports two voting modes:

### 1. Anonymous Voting (Default)
- Agents see anonymous IDs: `agent1`, `agent2`, `agent3`, etc.
- Real identities are hidden to avoid bias based on model names
- Promotes objective evaluation based on answer quality

### 2. Identified Voting
- Agents see real agent IDs (such as model names: `claude3.5haiku`, `gemini2.5flash`, `gpt5nano`, etc.)
- May vote based on model reputation or historical performance
- May affect self-voting tendencies and vote/improve ratio

## 🚀 Usage Methods

### Command Line Method

#### Anonymous Voting (Default)
```bash
# Using configuration file
uv run python -m massgen.cli --config config.yaml "Your question"

# Quick setup
uv run python -m massgen.cli --backend openai --model gpt-4o-mini "Your question"
```

#### Identified Voting
```bash
# Using --identified-voting flag
uv run python -m massgen.cli --config config.yaml --identified-voting "Your question"

# Quick setup
uv run python -m massgen.cli --backend openai --model gpt-4o-mini --identified-voting "Your question"
```

### Configuration File Method

Set the `orchestrator.anonymous_voting` parameter in the configuration file:

```yaml
# Anonymous voting (default)
orchestrator:
  anonymous_voting: true  # Or omit this line

# Identified voting
orchestrator:
  anonymous_voting: false
```

## 🔬 Experiments

### Objective
Compare the behavioral patterns of anonymous vs. identified voting mechanisms in multi-agent coordination systems.

### Research Questions
1. How does agent identity visibility affect voting distribution patterns across agents?
2. Does identity visibility increase or decrease self-voting frequency?
3. How does the vote/improvement ratio differ between anonymous and identified voting modes?
4. Which voting mechanism achieves consensus more reliably and in fewer rounds?
5. Do voting reasons differ in objectivity and identity-related content between the two modes?
6. Which voting mechanism produces statistically higher answer accuracy?

### Note on Counterfactual Analysis
While comparing identical answers under different voting modes would provide ideal controlled experiments, this analysis is not currently implemented. The current approach focuses on statistical comparison across different runs, as agent responses are non-deterministic. Future versions may include counterfactual analysis options for more rigorous experimental control.

### Key Metrics
1. **Voting Distribution**: Count of votes allocated to each agent in each voting round
2. **Self-Voting Frequency**: Percentage of rounds where agents vote for their own answers for each agent
3. **Vote/Improvement Ratio**: Ratio of voting decisions to improvement decisions for each agent
4. **Consensus Round**: Round number when consensus is reached
5. **Voting Reasons**: Text analysis of voting rationales for bias indicators and identity references
6. **Answer Quality**: Statistical comparison of final answer quality between voting mechanisms
