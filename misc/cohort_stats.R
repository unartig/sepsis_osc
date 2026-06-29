library(readr)
library(ricu)
library(dplyr)
library(tidyr)
library(data.table)

# ==============================================================================
# 1. THE SINGLE SWITCH: Change this to "miiv" or "eicu"
# ==============================================================================
target_src <- "miiv" 

# Dynamic file path selection based on your target source
csv_file <- paste0("cohort_ids_", target_src, ".csv")

# ==============================================================================
# 2. LOAD COHORT AND INITIALIZE RICU
# ==============================================================================
df <- read_csv(csv_file, col_types = cols(stay_id = col_character())) |>
  select(stay_id)

id_tbl <- as_id_tbl(df, id_vars = "stay_id")
# Dynamically register the ricu target structure class
class(id_tbl) <- c(paste0(target_src, "_id_tbl"), class(id_tbl))

attach_src(target_src)

# ==============================================================================
# 3. LOAD CORE CLINICAL CONCEPTS (Database-Agnostic)
# ==============================================================================
demo_concepts <- load_concepts(
  c("los_hosp", "death"), 
  src = target_src, 
  id_tbl = id_tbl
) |> as_tibble()

# Standardize the output identifier column back to a plain string map
colnames(demo_concepts)[1] <- "join_stay_id"
demo_concepts <- demo_concepts |> mutate(join_stay_id = as.character(join_stay_id))

# ==============================================================================
# 4. DYNAMIC ENVIRONMENT COLUMN MAPPING
# ==============================================================================
# Unpack the raw table data safely based on which database is active
if (target_src == "miiv") {
  
  # MIMIC-IV pulls stay mappings from icustays, joined to admissions
  icu_tab <- as.data.table(miiv$icustays) |> as_tibble()
  adm_tab <- as.data.table(miiv$admissions) |> as_tibble()
  
  raw_demographics <- icu_tab |>
    inner_join(adm_tab, by = c("subject_id", "hadm_id")) |>
    mutate(
      join_stay_id   = as.character(stay_id),
      admission_raw  = as.character(admission_type),
      ethnicity_raw  = as.character(race)
    ) |>
    select(join_stay_id, admission_raw, ethnicity_raw)
  
} else {
  
  # eICU pulls demographics directly out of the patient table
  patient_tab <- as.data.table(eicu$patient) |> as_tibble()
  
  raw_demographics <- patient_tab |>
    mutate(
      join_stay_id   = as.character(patientunitstayid),
      admission_raw  = as.character(hospitaladmitsource),
      ethnicity_raw  = as.character(ethnicity)
    ) |>
    select(join_stay_id, admission_raw, ethnicity_raw)
}

# ==============================================================================
# 5. COMBINE AND STANDARDIZE TEXT VALUES
# ==============================================================================
yaib_cohort <- id_tbl |>
  as_tibble() |>
  left_join(demo_concepts, by = c("stay_id" = "join_stay_id")) |>
  left_join(raw_demographics, by = c("stay_id" = "join_stay_id")) |>
  mutate(
    # 'los_hosp' is standardized by ricu in days; convert to hours
    hospital_los_hours = los_hosp * 24,
    
    # 'death' is standardized by ricu as a boolean logical flag
    hospital_expire_flag = if_else(death == TRUE, 1, 0, missing = 0),
    
    # Cross-database harmonized admission classification
    admission_group = case_when(
      tolower(admission_raw) %in% c("emergency", "urgent", "ew emer.", "direct emer.", "emergency room", "emergency department") ~ "Medical",
      tolower(admission_raw) == "elective" | grepl("operating room|recovery|or", tolower(admission_raw)) ~ "Surgical",
      TRUE ~ "Other/Unknown"
    ),
    
    # Cross-database harmonized ethnicity classification
    ethnicity_group = case_when(
      tolower(ethnicity_raw) %in% c("white", "caucasian") ~ "White",
      tolower(ethnicity_raw) %in% c("black/african american", "african american", "black") ~ "Black",
      tolower(ethnicity_raw) %in% c("asian", "asian - asian indian") ~ "Asian",
      tolower(ethnicity_raw) %in% c("hispanic or latino", "hispanic/latino", "hispanic") ~ "Hispanic",
      TRUE ~ "Other/Unknown"
    )
  ) |>
  select(stay_id, race = ethnicity_raw, admission_type = admission_raw, 
         hospital_expire_flag, hospital_los_hours, ethnicity_group, admission_group)

print(yaib_cohort)

# ==============================================================================
# 6. SUMMARY STATS AND EXPORT
# ==============================================================================
summary_stats <- yaib_cohort |>
  summarise(
    median_hospital_los = median(hospital_los_hours, na.rm = TRUE),
    iqr_25 = quantile(hospital_los_hours, 0.25, na.rm = TRUE),
    iqr_75 = quantile(hospital_los_hours, 0.75, na.rm = TRUE),
    mortality_rate = mean(hospital_expire_flag, na.rm = TRUE)
  )

print(summary_stats)

write_csv(yaib_cohort, "cohort_stats.csv")
