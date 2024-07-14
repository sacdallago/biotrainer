# Contributing to biotrainer

First off, a very warm thank you for considering contributing to `biotrainer`! It is people like you, 
living the spirit of open source software, that will make a difference.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching on GitHub
  under [Issues](https://github.com/sacdallago/biotrainer/issues).
- If you're unable to find an open issue addressing the
  problem, [open a new one](https://github.com/sacdallago/biotrainer/issues/new). Be sure to include a title and clear
  description, as much relevant information as possible, and a code sample or an executable test case demonstrating the
  expected behavior that is not occurring.

### Suggesting Enhancements

- Open a new issue with a clear title and detailed description of the suggested enhancement.

## GitFlow Workflow

At first, please create a fork of the repository in your own account. All PRs will be re-based on the development
branch of this main repository (see below).
We use a modified GitFlow workflow for this project. Here's an overview of the process:

### Main Branches

- `main`: This branch contains production-ready code. All releases are merged into `main` and tagged with a version
  number.
- `develop`: This is our main development branch. All features and non-emergency fixes are merged here.

### Supporting Branches

- Feature Branches:
    - Name format: `feature/your-feature-name`
    - Branch off from: `develop`
    - Merge back into: `develop`
    - Used for developing new features or enhancements.

- Bugfix Branches:
    - Name format: `bugfix/issue-description`
    - Branch off from: `develop`
    - Merge back into: `develop`
    - Used for fixing non-critical bugs.

- Release Branches:
    - Name format: `release/vX-Y-Z`
    - Branch off from: `develop`
    - Merge back into: `develop` and `main`
    - Used for preparing a new production release.

- Hotfix Branches:
    - Name format: `hotfix/issue-description`
    - Branch off from: `main`
    - Merge back into: `develop` and `main`
    - Used for critical bugfixes that need to be addressed immediately.

### Workflow Steps

1. For a new feature or non-critical bug fix:
    - Create a new feature or bugfix branch from `develop`.
    - Work on your changes in this branch.
    - When ready, create a pull request to merge your branch into `develop`.

2. For preparing a release:
    - Create a release branch from `develop`.
    - Make any final adjustments, version number updates, etc.
    - Create a pull request to merge the release branch into `main`.
    - After merging into `main`, also merge back into `develop`.
    - Tag the merge commit in `main` with the version number.

3. For critical hotfixes:
    - Create a hotfix branch from `main`.
    - Make your fixes.
    - Create a pull request to merge into `main`.
    - After merging into `main`, also merge into `develop`.
    - Tag the merge commit in `main` with an updated version number.

### Pull Requests

1. Ensure your code adheres to the project's coding standards.
2. Update the README.md with details of changes to the interface, if applicable.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would
   represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of one other developer, or if you do not have
   permission to do that, you may request the reviewer to merge it for you.

## Styleguides

### Git Commit Messages

- Use the present tense ("Adding feature" not "Added feature")
- Limit the first line to 72 characters or fewer
- Reference issues and pull requests liberally after the first line

### Python Styleguide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- Use 4 spaces for indentation (not tabs).
- Use docstrings for functions, classes, and modules.

### Documentation Styleguide

- Use [Markdown](https://daringfireball.net/projects/markdown/) for documentation.

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

* `bug` - Issues that are bugs.
* `enhancement` - Issues that are feature requests.
* `documentation` - Issues or pull requests related to documentation.
* `refactoring` - If you change something in the code, e.g. renaming function names.
* `maintenance` -  If you update parts of the project to newer versions, e.g. dependency updates or fixing examples.
* `good first issue` - Good for newcomers.

Thank you for contributing to `biotrainer`!
