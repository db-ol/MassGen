# for i in {0..2}; do
#   uv run python -m massgen.cli --config four_agent_test.yaml --question-number $i
# done


# How many jobs you want at the same time
CONCURRENCY=3

for i in 6 14; do
  echo $i
done | xargs -n1 -P"$CONCURRENCY" -I{} \
  uv run python -m massgen.cli --config four_agent_test.yaml --question-number {} --identified-voting

# uv run python -m massgen.cli --config four_agent_test.yaml --question-number 10


# ╔══ 🎯 FINAL COORDINATED ANSWER ══╗
# ║ 🏆 Selected agent: gemini2.5pro ║
# ╚═════════════════════════════════╝
# ╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
# │ ✅ Selected by: gemini2.5pro                                                                                                               │
# │ 🗳️ Vote results: gemini2.5pro: 2, gpt5nano: 1             