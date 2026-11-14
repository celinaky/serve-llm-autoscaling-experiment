#!/usr/bin/env python3
"""Compatibility entry point for the packaged session_lanes plot."""
import runpy

if __name__ == "__main__":
    runpy.run_module("serve_llm_autoscaling.diagnostics.session_lanes", run_name="__main__")
