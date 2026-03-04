# -*- coding: utf-8 -*-
"""
WirelessBench Data Preprocessing Module
========================================

Tools for building, expanding, cleaning, and analyzing the WirelessBench dataset.

Submodules:
  - expand_dataset   : LLM-powered dataset expansion with domain rotation
  - format_and_split : Normalize field order, translate Chinese, split test/validate
  - txt_to_jsonl     : Convert raw text files to JSONL with JSON error recovery
  - visualize        : Dataset analysis & visualization (knowledge points, difficulty)
  - llm_batcher      : Concurrent LLM API caller with retry logic
"""
