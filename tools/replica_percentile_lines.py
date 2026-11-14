#!/usr/bin/env python3
"""Compatibility entry point for the packaged replica_percentile_lines plot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("serve_llm_autoscaling.diagnostics.replica_percentile_lines", run_name="__main__")
