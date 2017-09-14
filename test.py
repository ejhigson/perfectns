#!/usr/bin/python
"""Contains the functions which perform nested sampling given input from settings. These are all called from within the wrapper function nested_sampling(settings)."""

import npns.nested_sampling as ns
import settings_npns
settings = settings_npns.NestedSamplingSettings()


r = ns.generate_standard_run(10, settings)
