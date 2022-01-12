# This script is adapted from python-semantic-release
# flake8: noqa

import sys


def patch(required_version):
    """
    Write the new version to init and test file
    """

    # Edit src
    with open('src/tf_transformers/__init__.py', 'w') as f:
        f.write('__version__ = "{}"\n'.format(required_version))

    # Edit test
    test_content = 'from tf_transformers import __version__\n\nversion = "{}"\ndef test_version():\n    assert __version__ == version\n'
    with open('tests/test_tf_transformers.py', 'w') as f:
        f.write(test_content.format(required_version))


if __name__ == '__main__':
    version = sys.argv[-1]  # last cli argument
    patch(version)
