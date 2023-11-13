#!/bin/bash
#python3 -m venv venv
#source venv/bin/activate
pip3 install -r requirements.txt

cat <<EOF >key.json
{
  "openai": {
    "organisation": "",
    "api_key": ""
  },
  "anthropic": {
    "api_key": ""
  },
  "alephalpha": {
    "api_key": ""
  },
  "huggingface": {
    "api_key": ""
  }
}
EOF

echo "Please add the keys to the key.json manually."
