# HIV Treatment Adherence Analysis
# ================================
# Survival analysis of factors affecting antiretroviral therapy adherence
# using data from a Nigerian HIV treatment program.
#
# Author: Dr. Charles Osahenrumwen Omorodion
# Date: April 2026

# Load required libraries
library(survival)
library(survminer)
library(tidyverse)
library(ggplot2)
library(coxphw)

# Set theme for visualizations
theme_set(theme_minimal())

# ============================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================

load_hiv_data <- function(filepath) {
  #' Load and prepare HIV treatment data
  #'
  #' @param filepath Path to the CSV data file
  #' @return Prepared dataframe
  
  cat("Loading HIV treatment data...\n")
  
  data <- read_csv(filepath) %>%
    mutate(
      # Convert dates
      enrollment_date = as.Date(enrollment_date),
      event_date = as.Date(event_date),
      
      # Create survival time in months
      survival_time = as.numeric(difftime(event_date, enrollment_date, units = "days")) / 30.44,
      
      # Event indicator (1 = treatment discontinuation, 0 = censored)
      event = as.numeric(discontinued),
      
      # Factor variables
      gender = factor(gender, levels = c("M", "F"), labels = c("Male", "Female")),
      baseline_cd4_cat = cut(baseline_cd4, 
                             breaks = c(0, 200, 350, 500, Inf),
                             labels = c("<200", "200-349", "350-499", "≥500")),
      age_group = cut(age, 
                      breaks = c(0, 25, 35, 45, 55, Inf),
                      labels = c("<25", "25-34", "35-44", "45-54", "55+")),
      adherence_support = factor(adherence_support),
      side_effects = factor(side_effects),
      distance_to_clinic = factor(distance_to_clinic,
                                  levels = c("<5km", "5-15km", ">15km"))
    )
  
  cat(sprintf("Loaded %d patient records\n", nrow(data)))
  cat(sprintf("Events (discontinuations): %d (%.1f%%)\n", 
              sum(data$event), mean(data$event) * 100))
  
  return(data)
}

# ============================================================
# 2. DESCRIPTIVE ANALYSIS
# ============================================================

descriptive_analysis <- function(data) {
  #' Perform descriptive analysis of the cohort
  
  cat("\n" + rep("=", 50) + "\n")
  cat("DESCRIPTIVE ANALYSIS\n")
  cat(rep("=", 50) + "\n")
  
  # Summary statistics
  cat("\nCohort Characteristics:\n")
  summary_stats <- data %>%
    summarise(
      n = n(),
      age_mean = mean(age, na.rm = TRUE),
      age_sd = sd(age, na.rm = TRUE),
      female_pct = mean(gender == "Female") * 100,
      baseline_cd4_mean = mean(baseline_cd4, na.rm = TRUE),
      survival_time_median = median(survival_time, na.rm = TRUE)
    )
  print(summary_stats)
  
  # Distribution by key variables
  cat("\nGender Distribution:\n")
  print(table(data$gender))
  
  cat("\nAge Group Distribution:\n")
  print(table(data$age_group))
  
  cat("\nBaseline CD4 Distribution:\n")
  print(table(data$baseline_cd4_cat))
  
  cat("\nDistance to Clinic:\n")
  print(table(data$distance_to_clinic))
}

# ============================================================
# 3. SURVIVAL ANALYSIS
# ============================================================

fit_kaplan_meier <- function(data) {
  #' Fit Kaplan-Meier survival curves
  
  cat("\n" + rep("=", 50) + "\n")
  cat("KAPLAN-MEIER SURVIVAL ANALYSIS\n")
  cat(rep("=", 50) + "\n")
  
  # Overall survival
  km_fit <- survfit(Surv(survival_time, event) ~ 1, data = data)
  cat("\nOverall Survival Summary:\n")
  print(km_fit)
  
  # Median time to discontinuation
  cat(sprintf("\nMedian time to discontinuation: %.1f months\n", 
              km_fit$time[which.min(abs(km_fit$surv - 0.5))]))
  
  # Survival by gender
  km_gender <- survfit(Surv(survival_time, event) ~ gender, data = data)
  
  # Log-rank test
  logrank_test <- survdiff(Surv(survival_time, event) ~ gender, data = data)
  cat("\nLog-rank test (Gender):\n")
  print(logrank_test)
  
  # Plot survival curves
  p <- ggsurvplot(
    km_gender,
    data = data,
    pval = TRUE,
    conf.int = TRUE,
    risk.table = TRUE,
    xlab = "Time (months)",
    ylab = "Treatment Retention Probability",
    title = "Kaplan-Meier Survival Curves by Gender",
    legend.title = "Gender",
    palette = c("#E7B800", "#2E9FDF")
  )
  
  ggsave("km_survival_by_gender.png", p$plot, width = 10, height = 6, dpi = 300)
  cat("\nSurvival plot saved as 'km_survival_by_gender.png'\n")
  
  return(list(km_fit = km_fit, km_gender = km_gender))
}

fit_cox_model <- function(data) {
  #' Fit Cox proportional hazards model
  
  cat("\n" + rep("=", 50) + "\n")
  cat("COX PROPORTIONAL HAZARDS MODEL\n")
  cat(rep("=", 50) + "\n")
  
  # Fit full model
  cox_model <- coxph(
    Surv(survival_time, event) ~ 
      age + 
      gender + 
      baseline_cd4 +
      adherence_support +
      side_effects +
      distance_to_clinic +
      pill_burden,
    data = data
  )
  
  cat("\nCox Model Summary:\n")
  print(summary(cox_model))
  
  # Extract hazard ratios
  hr_table <- data.frame(
    Variable = names(coef(cox_model)),
    HR = exp(coef(cox_model)),
    CI_lower = exp(confint(cox_model))[, 1],
    CI_upper = exp(confint(cox_model))[, 2],
    p_value = summary(cox_model)$coefficients[, 5]
  )
  
  cat("\nHazard Ratios:\n")
  print(hr_table)
  
  # Test proportional hazards assumption
  cat("\nTesting Proportional Hazards Assumption:\n")
  zph_test <- cox.zph(cox_model)
  print(zph_test)
  
  # Plot Schoenfeld residuals
  png("schoenfeld_residuals.png", width = 1000, height = 800, res = 150)
  plot(zph_test)
  dev.off()
  cat("\nSchoenfeld residuals plot saved\n")
  
  # Forest plot of hazard ratios
  p <- ggforest(cox_model, data = data,
                main = "Hazard Ratios for Treatment Discontinuation")
  ggsave("cox_forest_plot.png", p, width = 12, height = 8, dpi = 300)
  cat("Forest plot saved as 'cox_forest_plot.png'\n")
  
  return(cox_model)
}

# ============================================================
# 4. RISK STRATIFICATION
# ============================================================

create_risk_stratification <- function(data, cox_model) {
  #' Create risk stratification tool
  
  cat("\n" + rep("=", 50) + "\n")
  cat("RISK STRATIFICATION TOOL\n")
  cat(rep("=", 50) + "\n")
  
  # Calculate risk scores
  data$risk_score <- predict(cox_model, type = "risk")
  
  # Create risk categories
  data$risk_category <- cut(
    data$risk_score,
    breaks = quantile(data$risk_score, probs = c(0, 0.33, 0.67, 1)),
    labels = c("Low", "Moderate", "High"),
    include.lowest = TRUE
  )
  
  # Survival by risk category
  km_risk <- survfit(Surv(survival_time, event) ~ risk_category, data = data)
  
  cat("\nSurvival by Risk Category:\n")
  print(km_risk)
  
  # Plot
  p <- ggsurvplot(
    km_risk,
    data = data,
    pval = TRUE,
    conf.int = TRUE,
    xlab = "Time (months)",
    ylab = "Treatment Retention Probability",
    title = "Survival Curves by Risk Category",
    legend.title = "Risk Category",
    palette = c("#2E9FDF", "#E7B800", "#FC4E07")
  )
  
  ggsave("km_survival_by_risk.png", p$plot, width = 10, height = 6, dpi = 300)
  cat("\nRisk stratification plot saved as 'km_survival_by_risk.png'\n")
  
  # Risk category distribution
  cat("\nRisk Category Distribution:\n")
  print(table(data$risk_category))
  
  return(data)
}

# ============================================================
# 5. MAIN EXECUTION
# ============================================================

main <- function() {
  #' Main execution function
  
  cat("HIV Treatment Adherence Analysis\n")
  cat("================================\n")
  cat("\nThis analysis examines factors affecting antiretroviral therapy\n")
  cat("adherence using survival analysis techniques.\n")
  cat("\nMethods:\n")
  cat("- Kaplan-Meier survival curves\n")
  cat("- Cox proportional hazards modeling\n")
  cat("- Risk stratification\n")
  
  # Note: In practice, load actual data
  # data <- load_hiv_data("hiv_cohort_data.csv")
  # descriptive_analysis(data)
  # km_results <- fit_kaplan_meier(data)
  # cox_model <- fit_cox_model(data)
  # data_with_risk <- create_risk_stratification(data, cox_model)
  
  cat("\n" + rep("=", 50) + "\n")
  cat("Note: To run full analysis, provide path to HIV cohort dataset.\n")
  cat("Expected columns: patient_id, enrollment_date, event_date,\n")
  cat("discontinued, age, gender, baseline_cd4, adherence_support,\n")
  cat("side_effects, distance_to_clinic, pill_burden\n")
}

# Run main function
if (interactive()) {
  main()
}
