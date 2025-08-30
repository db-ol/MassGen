# for i in {0..2}; do
#   uv run python -m massgen.cli --config four_agent_test.yaml --question-number $i
# done


# How many jobs you want at the same time
CONCURRENCY=8

nums=(5 7 11 13 15 17 19 20 21 23 24 27 28 33 34 38 39 40 43 44 45 47 48 49 51 53 54 55 56 57 58 59 62 63 68 69 70 73 77 78 79 82 83 84 87 89 92 93 94 95 96 98 102 106 109 112 113 114 120 123 125 127 131 134 136 141 145 146 154 165 172 177 184 188)

printf '%s\n' "${nums[@]}" | xargs -n1 -P"$CONCURRENCY" -I{} \
  uv run python -m massgen.cli --config four_agent_test.yaml --question-number {} --identified-voting


# uv run python -m massgen.cli --config four_agent_test.yaml --question-number 10


# ╔══ 🎯 FINAL COORDINATED ANSWER ══╗
# ║ 🏆 Selected agent: gemini2.5pro ║
# ╚═════════════════════════════════╝
# ╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
# │ ✅ Selected by: gemini2.5pro                                                                                                               │
# │ 🗳️ Vote results: gemini2.5pro: 2, gpt5nano: 1             