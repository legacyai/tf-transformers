# This script is adapted from python-semantic-release

from absl import logging
from semantic_release import cli

logging.set_verbosity("DEBUG")


def push_version(retry=False, noop=False, force_level=None, **kwargs):
    """
    Detect the new version according to git log and semver.
    Write the new version number and commit it, unless the noop option is True.
    """

    if retry:
        logging.info("Retrying publication of the same version")
    else:
        logging.info("Creating new version")

    # Get the current version number
    try:
        current_version = cli.get_current_version()
        logging.info(f"Current version: {current_version}")
    except cli.GitError as e:
        logging.error(str(e))
        return False

    if not cli.should_bump_version(
        current_version=current_version, new_version=current_version, retry=retry, noop=noop
    ):
        return False

    if retry:
        # No need to make changes to the repo, we're just retrying.
        return True

    # Bump the version
    cli.bump_version(current_version, level_bump='dummy')
    return True


if __name__ == '__main__':
    push_version()
