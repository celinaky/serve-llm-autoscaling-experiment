#!/usr/bin/env python3
"""Compatibility entry point for the packaged policy_vs_engine_mean plot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("serve_llm_autoscaling.diagnostics.policy_vs_engine_mean", run_name="__main__")
