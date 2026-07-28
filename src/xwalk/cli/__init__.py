"""The xwalk command-line interface. A thin shell over the SDK.

Deliberately does not import `main` here: an eager re-export makes
`python -m xwalk.cli.main` emit a runpy double-import warning. Use
`python -m xwalk.cli`, the `xwalk` console script, or import from
`xwalk.cli.main` directly.
"""
