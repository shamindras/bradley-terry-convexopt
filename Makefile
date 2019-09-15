.PHONY : conda_prod1 conda_rem_reinst_prod1 get_nascar_data get_nfl_filt_data clean

# Create the various conda environments
conda_prod1:
	conda env create -f=./conda_envs/sklearnprod0/environment.yml

conda_rem_reinst_prod1:
	conda remove --name sklearnprod1 --all
	conda_prod1

get_nascar_data:
	Rscript R/get_nascar_data.R
	Rscript R/get_nascar_ranking_data.R

get_nfl_filt_data:
	Rscript R/get_nfl_data_filt_teams.R

clean:
	find R  -iname '*.DS_Store' -print0 | xargs -0 rm -rf
	find R  -iname '.Rhistory' -print0 | xargs -0 rm -rf
	find R  -iname '*.nb.html' -print0 | xargs -0 rm -rf
	find R  -iname '*_files' -print0 | xargs -0 rm -rf
	find R  -iname '*_cache' -print0 | xargs -0 rm -rf
