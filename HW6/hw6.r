library(tidyverse)
library(tidymodels)
library(rpart.plot)
library(vip)
library(janitor)
library(randomForest)
library(xgboost)
library(vroom)
library(corrplot)
library(ranger)
tidymodels_prefer()

pokemon <- vroom("data/Pokemon.csv")
pokemon <- clean_names(pokemon)
head(pokemon)

pokemon_cleaned <- pokemon %>%
    filter(type_1 %in% c("Bug", "Fire", "Grass", "Normal", "Water", "Psychic"))

pokemon_cleaned$type_1 <- factor(pokemon_cleaned$type_1)
pokemon_cleaned$legendary <- factor(pokemon_cleaned$legendary, 
    c("TRUE", "FALSE"))

levels(pokemon_cleaned$type_1)
levels(pokemon_cleaned$legendary)

pokemon_split <- initial_split(pokemon_cleaned, strata = type_1)
pokemon_training <- training(pokemon_split)
pokemon_testing <- testing(pokemon_split)

pokemon_folded <- pokemon_training %>%
    vfold_cv(, v = 5, strata = type_1)

pokemon_recipe <- pokemon_training %>%
    recipe(type_1 ~ legendary +
        generation +
        sp_atk +
        attack +
        speed +
        defense +
        hp +
        sp_def) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_normalize(all_predictors())

pokemon_cleaned %>%
    select_if(is.numeric) %>%
    select(-c(number)) %>%
    cor() %>%
    corrplot(method = "number")


# Number removed, they have approximated 0 correlation intended by Dev team to achieve balance

# since Tanky pokemon usually have nerfed speed
# HP/Def/Sp_def and speed have low correlation

# Tanky pokemon often has both high def and high special def, thus higher correlation

# For pokemons specialized in speical abilities (psychic especially)
# Special attack has the highest correlation with special def

# (If We have dragon type the correlation will be slightly different)

tree_spec <- decision_tree() %>%
    set_engine("rpart") %>%
    set_mode("classification")

tree_tune <- tree_spec %>%
    set_args(cost_complexity = tune())

tree_workflow <- workflow() %>%
    add_model(tree_tune) %>%
    add_recipe(pokemon_recipe)

param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

tune_res <- tune_grid(
    tree_workflow,
    resamples = pokemon_folded,
    grid = param_grid,
    metric = metric_set(roc_auc)
)

tune_res %>%
    autoplot()

# Higher cost complexity is better for a single tree

collect_metrics(tune_res) %>%
    arrange(desc(mean))

best_complexity <- select_best(tune_res, metric = "roc_auc")

tree_final <- finalize_workflow(tree_workflow, best_complexity)

tree_final_fit <- fit(tree_final, data = pokemon_training)

tree_final_fit %>%
    extract_fit_engine() %>%
    rpart.plot()

rf_spec <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
    set_engine("ranger", importance = "impurity") %>%
    set_mode("classification")

rf_workflow <- workflow() %>%
    add_model(rf_spec) %>%
    add_recipe(pokemon_recipe)

rf_param_grid <- grid_regular(
    mtry(c(1,5)),
    trees(c(1, 1000)),
    min_n(c(2, 40)),
    levels = 10
)


# Mtry: number of predictors sampled every time R split a tree
# Tree: number of trees to construct
# Min_n number of data needed to split a node

# Mtry is selecting predictors so =8 is when selecting all the predictors,
# just like a single decision tree

rf_tune_res <- tune_grid(
    rf_workflow,
    resamples = pokemon_folded,
    grid = rf_param_grid,
    metric = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)
)

rf_tune_res %>%
    autoplot()

# 3 mtry 334 trees and 23 min_n

rf_tune_res %>%
    collect_metrics() %>%
    arrange(desc(mean))

rf_best_tune <- select_best(rf_tune_res, metric = "roc_auc")
rf_final <- finalize_workflow(rf_workflow, rf_best_tune)
rf_final_fit <- fit(rf_final, data = pokemon_training)
rf_final_fit

vip(rf_final_fit %>% extract_fit_parsnip())

