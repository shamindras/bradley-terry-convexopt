.PHONY : conda_prod0 conda_prod1 clean

# Create the various conda environments
conda_prod0:
	conda env create -f=./conda_envs/sklearnprod0/environment.yml

conda_prod1:
	conda env create -f=./conda_envs/sklearnprod0/environment.yml

conda_rem_reinst_prod0:
	conda remove --name sklearnprod0 --all
	conda_prod0

conda_rem_reinst_prod1:
	conda remove --name sklearnprod1 --all
	conda_prod1

clean:
	find R  -iname '*.DS_Store' -print0 | xargs -0 rm -rf
	find R  -iname '.Rhistory' -print0 | xargs -0 rm -rf
	find R  -iname '*.nb.html' -print0 | xargs -0 rm -rf
	find R  -iname '*_files' -print0 | xargs -0 rm -rf
	find R  -iname '*_cache' -print0 | xargs -0 rm -rf
