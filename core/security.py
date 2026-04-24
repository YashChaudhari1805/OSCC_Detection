"""
core/security.py — NOTE
------------------------
This file is NOT imported by the FastAPI application (main.py).

Auth and rate-limiting are implemented directly in main.py using
FastAPI's Depends() system. The Flask-based decorators here
(@protected, @rate_limited) use `flask` imports that are NOT installed
and would cause an ImportError if used.

This file can be safely deleted. It is kept only for reference.
"""
# The original Flask decorators are intentionally not re-exported here.
# See main.py → rate_limit(), verify_key(), and _protected for the
# FastAPI equivalents.
