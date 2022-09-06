---
layout: page
author: Cem Kose
title: Conda CLI Cheat-Sheet
excerpt: Somewhat dated conda cli cheatsheet.
---


# Conda CLI Cheat-Sheet

---


## Overview

The <b>conda</b> command is the primary interface for managing installations of various packages.

Many frequently used command options that use <b>2 dashes</b> (--) can be <b>abbreviated</b> using <b>1 dash</b> and the <b>first letter</b> of the option.

<b>--envs -> -e --name -> -n</b>

---


## Managing Conda

### Get conda version:


> -- conda --version conda -V




### Update conda to the current version:

> -- conda update conda

---


## Managing Environments

Conda allows you to create separate environments containing <b>files</b>, <b>packages</b> and <b>dependencies</b> that do <b>not</b> interact with each other.

The default conda environment is named <b>base</b>. Keep programs <b>isolated</b> by creating <b>separate</b> environments.

---


### Creating a new environment and installing a package in it.

> -- conda create --name envname tensorflow

> -- conda create -n envname tensorflow

Conda checks to see what <b>additional packages</b> tensorflow will need, and asks if you want to proceed.

### To activate the new environment

> -- conda activate envname


### To see a list of all you Environments

> -- conda info --envs

> -- conda info -e

### To change current environment back to base

> -- conda activate

### To deactivate current environment

> -- conda deactivate

### Delete an environment

> -- conda env remove --name envname

---


## Managing Python

When a new environment is created conda <b>installs</b> the <b>same Python version</b> you used when you <b>downloaded and installed Anaconda</b>.

To use a <b>different</b> version of <b>Python</b> create a <b>new</b> environment and <b>specify</b> the <b>version</b> of <b>Python</b> that you want.

> -- conda create --name envname python=2.7

> -- conda create -n envname python=3.5

### Verify which version of Python is in your current environment

> -- python --version python -V

---

## Managing packages

### To check if a package is available from the anaconda repository

> -- conda search tensorflow

If conda displays a list of <b>packages</b> with that <b>name</b> you know that the package is <b>available</b> on the <b>Anaconda repository</b>.

### To install package into current environment

> -- conda install tensorflow

### To list all packages available in current environment

> -- conda list

---

## Sharing Environments

### Make an exact copy of an environment

> -- conda create --clone envname --name newenv

### Export an environment to a YAML file that can be read on Windows, macOS and Linux

> -- conda env export --name envname > envname.yml

### Create an environment from the file named environment.yml in the current directory

> -- conda env create

---

## Additional Useful Commands

### Detailed information about package version

> -- conda search pkgname --info

### Remove unused cached package version

> -- conda clean --packages

### Remove a package from an environment

> -- conda uninstall pkgname --name envname
