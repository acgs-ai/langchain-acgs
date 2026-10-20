#!/bin/bash
# Only langchain.agents (middleware) may be imported from the langchain package; everything else comes from
# langchain_core or langgraph.
set -eu
if grep -rnE --include=*.py "^\s*(from|import) langchain(\.|\s)" langchain_acgs | grep -vE "(from|import) langchain\.agents"; then
    echo "Import from langchain_core or langchain.agents instead." >&2
    exit 1
fi
