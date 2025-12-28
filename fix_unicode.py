#!/usr/bin/env python3
"""Fix Unicode characters in test file for Windows compatibility."""

import re

# Read the file
with open('test_advanced_features_simple.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters
replacements = {
    '✓': '[PASS]',
    '✗': '[FAIL]',
    '⚠️': '[WARN]',
    '❌': '[FAIL]',
    '🎉': '[SUCCESS]',
    '✅': '[PASS]'
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('test_advanced_features_simple.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Unicode characters in test file")