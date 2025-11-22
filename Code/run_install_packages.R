#All required packages
packages <- c(
  "quantmod", 
  "rugarch", 
  "xts", 
  "tibble",
  "dplyr", 
  "tidyr", 
  "lubridate", 
  "readxl", 
  "janitor",
  "zoo", 
  "here", 
  "writexl", 
  "readr",
  "tidyverse", 
  "glmnet", 
  "MASS", 
  "caret",
  "TTR", 
  "ggplot2", 
  "ranger", 
  "randomForest",
  "fastshap", 
  "shapviz"
)

# Find any not yet installed
missing <- packages[!packages %in% installed.packages()[, "Package"]]

# Install missing packages
if (length(missing) > 0) {
  install.packages(missing, dependencies = TRUE)
} else {
  message("All packages already installed.")
}

# Load all packages
invisible(lapply(packages, library, character.only = TRUE))
message("All packages loaded successfully!")
