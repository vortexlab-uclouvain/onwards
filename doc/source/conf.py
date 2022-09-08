# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OnWaRDS'
copyright = '2022, Maxime Lejeune UCLouvain'
author = 'Maxime Lejeune UCLouvain'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Core library for html generation from docstrings
    # 'sphinx.ext.autosummary',   # Create neat summary tables
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
]

templates_path = ['_templates', 'myst_parser']
exclude_patterns = []

source_suffix = ['.rst', '.md']

# autodoc_typehints = "none"
# autodoc_typehints = "description"
# autodoc_typehints_description_target = "documented"
# autodoc_typehints = "description"

# autodoc_default_options = {
#     'member-order': 'bysource',
#     # 'no-value': False,
#     # 'members':          True,
#     # 'undoc-members':    True,
#     'special-members': ['__init_states__', '__init_sensors__']
# # }
# autodoc_default_options = {
#     'member-order': 'bysource',
#     'autodoc_mock_imports' : ["lagSolver_c"],
# }

autoclass_content = 'both'
# napoleon_attr_annotations = True
# napoleon_use_ivar = False
# autodoc_typehints = "signature"

# def autodoc_skip_member(app, what, name, obj, skip, options):
#     exclusions = ('inherited_members')
#     exclude = options in exclusions
#     # return True if (skip or exclude) else None  # Can interfere with subsequent skip functions.
#     return True if exclude else None
    

# def setup(app):
#     app.connect("autodoc-skip-member", autodoc_skip_member)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
