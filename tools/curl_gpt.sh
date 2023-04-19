#!/bin/bash
OPENAI_API_KEY="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"
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "text-davinci-003",
    "prompt": "Say this is a test",
    "max_tokens": 7,
    "temperature": 0
  }'