import re

vote_recorded_re   = re.compile(r'(?P<src>[A-Za-z0-9_.-]+):\s*✅ Vote recorded for \[(?P<dst>[A-Za-z0-9_.-]+)\]')
answer_provided_re = re.compile(r'(?P<src>[A-Za-z0-9_.-]+):\s*✅ Answer provided\b')

a = "  • [14:04:57] 🔄 grok4: ✅ Answer provided"
m_ap = answer_provided_re.search(a)   # 用 search，不要 match
print(m_ap is not None, m_ap.group('src'))   # True  grok4

