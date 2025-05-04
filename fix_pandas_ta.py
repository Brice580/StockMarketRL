import sys
import os
import warnings

# First, suppress all DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Force disable the warnings in the logging module as well
import logging
logging.getLogger('pkg_resources').setLevel(logging.ERROR)

# Now, monkey patch the pkg_resources module to avoid the deprecation warning
try:
    import pkg_resources
    
    # Store the original warn function
    original_warn = warnings.warn
    
    # Define a patched warn function that ignores certain warnings
    def patched_warn(message, category=None, *args, **kwargs):
        if category == DeprecationWarning and "pkg_resources is deprecated" in str(message):
            return  # Skip this warning
        return original_warn(message, category, *args, **kwargs)
    
    # Replace the warnings.warn function with our patched version
    warnings.warn = patched_warn
    
    print("Successfully patched warnings for pandas_ta")
except Exception as e:
    print(f"Failed to patch warnings: {e}")

# Fix the numpy NaN issue
try:
    import numpy
    # Add NaN as an alias for nan
    numpy.NaN = numpy.nan
    print("Successfully patched numpy.NaN")
except Exception as e:
    print(f"Failed to patch numpy: {e}")

# Now we can safely import pandas_ta
try:
    import pandas_ta as ta
    print("Successfully imported pandas_ta")
except Exception as e:
    print(f"Failed to import pandas_ta: {e}") 