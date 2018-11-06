# The following script will give us the pairwise Bradley-Terry
# model data for NFL

#-------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------
# File for building WP model

# Access tidyverse:
# install.packages("tidyverse")
library(tidyverse)

# Access nflWAR:
# install.packages("devtools")
devtools::install_github("ryurko/nflWAR")
# library(nflWAR)

#-------------------------------------------------------------------
# Define Variables
#-------------------------------------------------------------------

all_seasons <- 2009:2016
nfl_csv_gh_ref <- "https://raw.github.com/ryurko/nflscrapR-data/master/legacy_data/season_games/games_"

#-------------------------------------------------------------------
# Get all win-loss data for all seasons
# Source: https://github.com/ryurko/nflscrapR-models/blob/master/R/init_models/init_wp_model.R#L25-L28
#-------------------------------------------------------------------

# Load and stack the games data for all specified seasons
games_data <-  all_seasons %>%
                    purrr::map_dfr(.,
                            function(x) {
                        suppressMessages(
                            readr::read_csv(paste(nfl_csv_gh_ref,
                                           x, ".csv", sep = "")))
})

#-------------------------------------------------------------------
# Basic checks on Data
#-------------------------------------------------------------------

# Just check the colnames for reference
base::colnames(games_data)

# We expect 256 rows of data per season
# - 32 teams playing 16 games each
# Note: Indeed this is the case!
# TODO: write a test
games_data %>%
    dplyr::group_by(Season) %>%
    dplyr::summarise(tot_played = n())

# We can also do this to get a crude average games per season
dim(games_data)[1]/length(all_seasons)

#-------------------------------------------------------------------
# Basic checks on Data
#-------------------------------------------------------------------

# Consider a specific (required) season
rseason <- 2016

# Get only the games for the required season
games_data_rseason <- games_data %>%
                        dplyr::filter(Season == rseason)


#-------------------------------------------------------------------
# CREATE CORE DATA for a SEASON
#-------------------------------------------------------------------

# We want to get an ordering of teams and a uniq team number for
# the season
unq_teams <- games_data_rseason %>%
                dplyr::select(home) %>%
                dplyr::distinct(.) %>%
                dplyr::rename(team = home) %>%
                dplyr::arrange(team) %>%
                dplyr::mutate(ind = 1:n())

View(unq_teams)

# Get Home teams as the primary "team" column
games_home <- games_data_rseason %>%
                dplyr::rename(team = home,
                              team_other = away,
                              team_score = homescore,
                              team_other_score = awayscore) %>%
                dplyr::select(GameID, date, team, team_other,
                              team_score, team_other_score,
                              Season)

# Get Away teams as the primary "team" column
games_away <- games_data_rseason %>%
                dplyr::rename(team = away,
                              team_other = home,
                              team_score = awayscore,
                              team_other_score = homescore) %>%
                dplyr::select(GameID, date, team, team_other,
                              team_score, team_other_score,
                              Season)

# Combine into a single dataframe and create the "round" for each team
# based on play date
games_all <- games_home %>%
                dplyr::bind_rows(games_away) %>%
                dplyr::group_by(team) %>%
                dplyr::mutate(round = row_number(date))

# Join on the unique team id number created for the "primary" team
games_all <- games_all %>%
                dplyr::left_join(x = ., y = unq_teams, by = c("team")) %>%
                dplyr::rename(team_ind = ind)

# Join on the unique team id number created for the "other" team
# diff indicator - currently gives 0 value for a tie
# TODO: Check how we will deal with ties in our model

games_all <- games_all %>%
                dplyr::left_join(x = ., y = unq_teams, by = c("team_other" = "team")) %>%
                dplyr::rename(team_other_ind = ind) %>%
                dplyr::mutate(diff = team_score - team_other_score,
                              diff_sign = sign(diff))

View(games_all)

#-------------------------------------------------------------------
# CREATE CARTESIAN pairwise master dataframe
#-------------------------------------------------------------------

all_rounds <- games_all %>%
                dplyr::ungroup() %>%
                dplyr::select(round) %>%
                dplyr::distinct() %>%
                dplyr::arrange(round)

all_team_ind <- unq_teams %>%
                    dplyr::ungroup() %>%
                    dplyr::select(ind) %>%
                    dplyr::distinct() %>%
                    dplyr::arrange(ind) %>%
                    dplyr::rename(team_ind = ind)

all_team_other_ind <- unq_teams %>%
                        dplyr::ungroup() %>%
                        dplyr::select(ind) %>%
                        dplyr::distinct() %>%
                        dplyr::arrange(ind) %>%
                        dplyr::rename(team_other_ind = ind)

# cartesian product
all_round_team_combs <- all_rounds %>%
                            tidyr::crossing(all_team_ind) %>%
                            tidyr::crossing(all_team_other_ind)

# Unit test: cartesian product - should be 0
# (just product of row dimensions less total dimension)
dim(all_rounds)[1] * dim(all_team_ind)[1] * dim(all_team_other_ind)[1] - dim(all_round_team_combs)[1]

#-------------------------------------------------------------------
# Create pairwise difference dataframes
#-------------------------------------------------------------------

# Now do it for a single round
round_num <- 1

# Get the diffs for a single round for all teams
games_round <- games_all %>%
                    dplyr::ungroup() %>%
                    dplyr::filter(round == round_num) %>%
                    dplyr::select(team_ind, team_other_ind, diff, diff_sign) %>%
                    dplyr::arrange(team_ind)

# Get a dataframe of all possible pairwise differences across teams
games_round_all_combs <- all_round_team_combs %>%
                            dplyr::ungroup() %>%
                            dplyr::filter(round == round_num) %>%
                            dplyr::select(team_ind, team_other_ind) %>%
                            dplyr::left_join(x = .,
                                             y = games_round,
                                             by = c("team_ind" = "team_ind",
                                                    "team_other_ind" = "team_other_ind"))
View(games_round_all_combs)

games_round_all_combs %>%
    dplyr::select(-diff_sign)