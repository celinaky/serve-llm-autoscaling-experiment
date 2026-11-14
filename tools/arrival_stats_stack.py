#!/usr/bin/env python3
"""Compatibility entry point for the packaged arrival_stats_stack plot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("serve_llm_autoscaling.diagnostics.arrival_stats_stack", run_name="__main__")
