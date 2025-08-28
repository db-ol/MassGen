# MassGen Voting Mechanism Comparison Guide

This guide provides detailed instructions on how to use MassGen's anonymous and non-anonymous voting features for experimental comparison.

## 🎭 Voting Mechanism Overview

MassGen supports two voting modes:

### 1. Anonymous Voting (Default)
- Agents see anonymous IDs: `agent1`, `agent2`, `agent3`, etc.
- Real identities are hidden to avoid bias based on model names
- Promotes objective evaluation based on answer quality

### 2. Non-Anonymous Voting
- Agents see real agent IDs (such as model names)
- May vote based on model reputation or historical performance
- Suitable for studying the impact of identity information on voting decisions

## 🚀 Usage Methods

### Command Line Method

#### Anonymous Voting (Default)
```bash
# Using configuration file
uv run python -m massgen.cli --config config.yaml "Your question"

# Quick setup
uv run python -m massgen.cli --backend openai --model gpt-4o-mini "Your question"
```

#### Non-Anonymous Voting
```bash
# Using --non-anonymous-voting flag
uv run python -m massgen.cli --config config.yaml --non-anonymous-voting "Your question"

# Quick setup
uv run python -m massgen.cli --backend openai --model gpt-4o-mini --non-anonymous-voting "Your question"
```

### Configuration File Method

Set the `orchestrator.anonymous_voting` parameter in the configuration file:

```yaml
# Anonymous voting (default)
orchestrator:
  anonymous_voting: true  # Or omit this line

# Non-anonymous voting
orchestrator:
  anonymous_voting: false
```

## 🔬 Experimental Design Suggestions

### Experiment 1: Voting Consistency Comparison
**Objective**: Compare agent voting consistency under anonymous and non-anonymous voting

**Steps**:
1. Use the same question and agent configuration
2. Run anonymous and non-anonymous voting separately
3. Record voting distribution and consensus achievement
4. Analyze differences in voting reasons

**Command Examples**:
```bash
# Anonymous voting
uv run python -m massgen.cli --config voting_comparison_example.yaml "Analyze future development trends of artificial intelligence"

# Non-anonymous voting
uv run python -m massgen.cli --config voting_comparison_example.yaml --non-anonymous-voting "Analyze future development trends of artificial intelligence"
```

### Experiment 2: Voting Reason Analysis
**Objective**: Analyze agent decision reasons under different voting modes

**Focus Points**:
- Whether voting reasons mention model identity
- Level of detail and objectivity in reasons
- Existence of reputation-based voting

### Experiment 3: Consensus Achievement Efficiency
**Objective**: Compare consensus achievement efficiency under two modes

**Metrics**:
- Number of voting rounds
- Time to reach consensus
- Frequency of vote changes

## 📊 Result Analysis

### Voting Result Display

#### Anonymous Voting Mode
```
🔀 Anonymous Agent Mapping:
   agent1 → gemini2.5flash
   agent2 → gpt5nano
   agent3 → claude3.5haiku
```

#### Non-Anonymous Voting Mode
```
🔍 Non-Anonymous Agent IDs:
   claude3.5haiku
   gemini2.5flash
   gpt5nano
```

### Key Metrics

1. **Voting Distribution**: Number of votes each agent receives
2. **Voting Reasons**: Specific reasons for agent voting
3. **Consensus Achievement**: Whether consensus is reached, which round
4. **Vote Changes**: Whether agents change their votes
