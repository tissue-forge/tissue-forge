Tissue Forge DevOps
====================

Tissue Forge development employs automated continuous integration and 
continuous delivery (CI/CD) to rapidly and reliably deliver the latest 
features for use by the general public. 
To download recent release and development builds and review build status 
by supported platform, visit the 
[Tissue Forge Azure project](https://dev.azure.com/Tissue-Forge/tissue-forge). 

Development and release of Tissue Forge is divided into three stages. 
The stage origin of an official Tissue Forge build is encoded in the 
header `tf_config.h` by a build qualifer (`TF_BUILDQUAL`).

* Develop
    * development stage
    * towards the next release
    * branch: [develop](https://github.com/tissue-forge/tissue-forge/tree/develop)
    * build qualifier: `develop`
* Staging
    * pre-release stage
    * validates packages for the next release
    * branch: [staging](https://github.com/tissue-forge/tissue-forge/tree/staging)
    * build qualifer: `staging`
* Version
    * release stage
    * new version packages distributed
    * branch: [main](https://github.com/tissue-forge/tissue-forge/tree/main) 
    * build qualifer: `release`

# Develop Stage #

Whether implementing new features or fixes, improving documentation or 
developing support for new languages and platforms, the 
[develop](https://github.com/tissue-forge/tissue-forge/tree/develop) branch 
of the Tissue Forge repository holds all revisions intended for the next 
version release with respect to the current version release, which is 
held in the [main](https://github.com/tissue-forge/tissue-forge) branch. 
A commit into [develop](https://github.com/tissue-forge/tissue-forge/tree/develop) 
triggers builds for local distributions on all supported platforms, 
which are publicly archived on success. 
Archived local builds include a supporting script to automate recreating 
the dependency environment needed to execute the pre-built binaries. 
The Tissue Forge version in the 
[develop](https://github.com/tissue-forge/tissue-forge/tree/develop) 
branch is the same as the current release, 
hence the build qualifier `develop` for version `X` states 
"developing from version `X`".

# Staging Stage #

All work in the [develop](https://github.com/tissue-forge/tissue-forge/tree/develop) 
branch is frozen and merged into the 
[staging](https://github.com/tissue-forge/tissue-forge/tree/staging) branch, where 
work is performed to validate all packages for distribution as the 
next release. 
All work in the [staging](https://github.com/tissue-forge/tissue-forge/tree/staging) 
branch is reserved for build- and distribution-specific fixes, which 
prohibits any work in the 
[develop](https://github.com/tissue-forge/tissue-forge/tree/develop) branch 
until all packages pass all tests. 
Any commits into the [staging](https://github.com/tissue-forge/tissue-forge/tree/staging) 
branch may be merged back into the 
[develop](https://github.com/tissue-forge/tissue-forge/tree/develop) branch for 
validation of local distributions. 
A commit into [staging](https://github.com/tissue-forge/tissue-forge/tree/staging) 
triggers builds for all distribution packages except local distributions, 
which are publicly archived on success.
The Tissue Forge version in the 
[staging](https://github.com/tissue-forge/tissue-forge/tree/staging) branch 
is the same as the current release, 
hence the build qualifier `staging` for version `X` states 
"staging development from version `X`".
Upon successful validation of all packages for distribution, 
development proceeds to the Version Stage.

# Version Stage #

Upon successful validation of all packages for distribution, 
all work in the 
[staging](https://github.com/tissue-forge/tissue-forge/tree/staging) branch 
is merged into the 
[main](https://github.com/tissue-forge/tissue-forge/tree/main) branch 
and the software version is incremented according to the content of the work. 
A commit into the [main](https://github.com/tissue-forge/tissue-forge/tree/main) branch 
triggers builds for all distribution packages, 
which are publicly archived and distributed on success.
The [main](https://github.com/tissue-forge/tissue-forge/tree/main) branch is then 
merged into the [develop](https://github.com/tissue-forge/tissue-forge/tree/develop) 
branch, and development returns to the Develop Stage. 
