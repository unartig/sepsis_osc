library(readr)
library(ricu)
library(dplyr)
library(tidyr)

df <- read_csv("cohort_ids.csv") |>
  select(stay_id) |>
  mutate(stay_id = as.character(stay_id))

id_tbl <- as_id_tbl(df, id_vars = "stay_id")
attach_src("miiv")

# Get ICU stay mapping
icu_mapping <- subset(miiv$icustays, stay_id %in% id_tbl$stay_id, 
                      c("subject_id", "hadm_id", "stay_id"))

# Get demographics AND hospital LOS times
demographics <- subset(miiv$admissions, 
                       subject_id %in% unique(icu_mapping$subject_id), 
                       c("subject_id", "hadm_id", "race", "admission_type", 
                         "hospital_expire_flag", "admittime", "dischtime"))

# Join mapping with demographics
demographics_with_stay <- merge(icu_mapping, demographics, 
                                by = c("subject_id", "hadm_id"))

# Convert and calculate hospital LOS
demo_wide <- demographics_with_stay |>
  as_tibble() |>
  mutate(stay_id = as.character(stay_id)) |>
  mutate(
    # Calculate hospital LOS in hours
    hospital_los_hours = as.numeric(difftime(dischtime, admittime, units = "hours")),
    admission_group = case_when(
      admission_type %in% c("EMERGENCY", "URGENT", "EW EMER.", "DIRECT EMER.") ~ "Medical",
      admission_type == "ELECTIVE" ~ "Surgical",
      TRUE ~ "Other/Unknown"
    ),
    ethnicity_group = case_when(
      race == "WHITE" ~ "White",
      race == "BLACK/AFRICAN AMERICAN" ~ "Black",
      race == "ASIAN" ~ "Asian",
      race %in% c("HISPANIC OR LATINO", "HISPANIC/LATINO") ~ "Hispanic",
      TRUE ~ "Other/Unknown"
    )
  ) |>
  select(stay_id, race, admission_type, hospital_expire_flag, 
         hospital_los_hours, ethnicity_group, admission_group)

# Join with original cohort
yaib_cohort <- id_tbl |>
  left_join(demo_wide, by = "stay_id")

print(yaib_cohort)

# Summary statistics to verify
yaib_cohort |>
  summarise(
    median_hospital_los = median(hospital_los_hours, na.rm = TRUE),
    iqr_25 = quantile(hospital_los_hours, 0.25, na.rm = TRUE),
    iqr_75 = quantile(hospital_los_hours, 0.75, na.rm = TRUE),
    mortality_rate = mean(hospital_expire_flag, na.rm = TRUE)
  )

write_csv(yaib_cohort, "cohort_stats.csv")
