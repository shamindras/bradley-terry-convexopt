# Time-varying Bradley Terry Model in Frequentist Setting

This repo stores the notes/ experiments and paper for
our Fall 2018 10725 project by Heejong Bong (HB), 
Wanshan Li (WL) and Shamindra Shrotriya (SS).

## Basic Setup & Installation

### Installing the `conda` environments

Firstly you need to [install Anaconda](https://www.continuum.io/downloads) on your computer

First fork and then clone the repo locally on your computer. Change directory to the repo folder.

To install the 3 conda environments just use the `Makefile` as shown below.

To create **all 3 environments** use:

```bash
make conda_all
```

Or to install them **individually** you can run the following commands separately

```bash
make conda_prod0
make conda_prod1
```

To confirm that the conda environments have installed correctly you can run the following in the terminal:
```bash
conda info -e
```

You should now see the 3 installed environments listed as required `sklearndev0`, `sklearndev1`, `sklearndev2` and `sklearnprod0`, `sklearnprod1`. 

To **activate** the conda environments simply use conda as usual i.e. `source activate sklearndev2`

### Quick info on each of the conda environments

`sklearnprod0`: This contains the latest conda **production** `scikit-learn` build and is useful for current production testing of `sklearn`

`sklearnprod1`: This contains the latest conda **production** `scikit-learn` build including `jupyter lab` which is important for our prototyping

## Git-Github - Typical Workflow

1. **First time only** Clone this repo through github i.e. `git clone git@github.com:shamindras/cmu_robustness_rg.git`

With the one-off setup complete we are ready to start coding! The main rule to remember is:

*Never commit/ merge new code **directly to master**!*

The typical workflow is as follows:

1. [Create](https://github.com/shamindras/cmu_robustness_rg/issues) a **github issue** for every task e.g. [example](https://github.com/shamindras/cmu_robustness_rg/issues/1).

* Add as many helpful links and details as possible.
* Typically I begin each issue with "FIX:"

e.g. **FIX: Write review of Shewchuck, Jonathan et. al paper** *Insert Issue details here*

2. Assign a person to the issue e.g. shamindras
3. Locally update master i.e.:

```bash
git checkout master
git pull origin master  # update local master
```

4. Create a **new branch** for the issue: `git checkout -b issue-**issue-number**-**short-description**`

e.g. `issue-01-shewchuck-conjugate-review` is one such typical branch name. Note you are now checked into the new branch and ready to go!

5. Do your great coding here :). Commit regularly with helpful messages e.g.:

*FIX: Issue #15, create first draft of review with sketch of key ideas*

6. Once you are ready to send a pull request you just commit and then run:

```bash
git checkout master
git pull origin master  # update local master
git checkout issue-**issue-number**-**short-description** # go back to working branch
git rebase master # sync branch with upstream master
```

7. Fix any merge conflicts and then commit branch
8. You can now commit the branch as `git push --set-upstream origin issue-**issue-number**-**short-description`
9. In the [origin github page](https://github.com/shamindras/cmu_robustness_rg) you will see the pull request.
10. Ask the reviewer to review as required. For all code review changes requested, just repeat **step 5** onwards until reviewer is satisfied
11. Once all changes are put in - the reviewer can merge the changes in upstream master
12. Then start a new issue from **Step 1** onwards!

### Members

The following members are contributing the repo

* Heejong Bong (HB)
* Wanshan Li (WL)
* Shamindra Shrotriya (SS)
