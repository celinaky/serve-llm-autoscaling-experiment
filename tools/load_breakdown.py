#!/usr/bin/env python3
"""Compatibility entry point for the packaged load_breakdown plot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("serve_llm_autoscaling.diagnostics.load_breakdown", run_name="__main__")
