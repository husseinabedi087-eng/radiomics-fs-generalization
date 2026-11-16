**Table S1. Included datasets (43).**

| Dataset | Group | n | p | log(p/n) | Minority-class count | Modality | Licence category | Flagged for S1 | Included in S2 |
|---|---|---|---|---|---|---|---|---|---|
| Ahn2021 | development | 114 | 108 | -0.054 | 51 | MRI | non_commercial | no | n/a |
| Arita2018 | development | 168 | 673 | 1.388 | 57 | MRI | open | yes | n/a |
| Brancato2023 | development | 58 | 2380 | 3.714 | 23 | MRI | open | no | n/a |
| Dai2023 | development | 119 | 851 | 1.967 | 31 | CT | open | no | n/a |
| HNSCC | development | 93 | 105 | 0.121 | 25 | CT | tcia_restricted_images_derived_features | no | n/a |
| Hosny2018B | development | 207 | 984 | 1.559 | 60 | CT | no_license_file_plos_data_policy | yes | n/a |
| ISPY1 | development | 161 | 370 | 0.832 | 69 | MRI | open | no | n/a |
| Meningioma-SEG-CLASS | development | 88 | 2030 | 3.138 | 38 | MRI | tcia_restricted_images_derived_features | no | n/a |
| Ramella2018 | development | 91 | 242 | 0.978 | 41 | CT | open | no | n/a |
| WORC-CRLM | development | 77 | 1015 | 2.579 | 37 | CT | custom_research_terms | no | n/a |
| WORC-Desmoid | development | 203 | 1015 | 1.609 | 72 | MRI | custom_research_terms | no | n/a |
| Wang2024 | development | 67 | 280 | 1.430 | 27 | MRI | open | yes | n/a |
| BraTS-2021 | confirmation | 577 | 4060 | 1.951 | 276 | MRI | custom_research_terms | no | no |
| C4KC-KiTS | confirmation | 70 | 315 | 1.504 | 24 | CT | open | no | yes |
| Dong2022 | confirmation | 279 | 851 | 1.115 | 136 | CT | open | no | yes |
| Fusco2022 | confirmation | 54 | 192 | 1.269 | 21 | MRI | open | no | yes |
| Granata2021 | confirmation | 88 | 579 | 1.884 | 41 | CT | open | no | yes |
| Head-Neck-PET-CT | confirmation | 91 | 210 | 0.836 | 30 | PET/CT | tcia_restricted_images_derived_features | no | yes |
| Head-Neck-Radiomics-HN1 | confirmation | 137 | 105 | -0.266 | 61 | CT | non_commercial | no | yes |
| Hosny2018A | confirmation | 293 | 984 | 1.211 | 134 | CT | no_license_file_plos_data_policy | yes | yes |
| Hosny2018C | confirmation | 183 | 984 | 1.682 | 50 | CT | no_license_file_plos_data_policy | yes | yes |
| Huang2023 | confirmation | 212 | 855 | 1.395 | 97 | CT | open | no | yes |
| Hunter2023 | confirmation | 520 | 1998 | 1.346 | 238 | CT | open | no | no |
| Keek2020 | confirmation | 273 | 1322 | 1.577 | 119 | CT | no_license_file_plos_data_policy | yes | no |
| LGG-1p19qDeletion | confirmation | 159 | 2030 | 2.547 | 57 | MRI | tcia_restricted_images_derived_features | no | no |
| LNDb | confirmation | 173 | 105 | -0.499 | 58 | CT | non_commercial | yes | yes |
| Lu2019 | confirmation | 75 | 657 | 2.170 | 20 | CT | open | no | yes |
| NSCLC-Radiogenomics | confirmation | 144 | 105 | -0.316 | 23 | PET/CT | open | no | yes |
| OcanaTienda2023 | confirmation | 67 | 1130 | 2.825 | 32 | MRI | open | no | no |
| PI-CAI | confirmation | 969 | 3045 | 1.145 | 332 | MRI | non_commercial | no | no |
| Petrillo2023 | confirmation | 128 | 851 | 1.894 | 47 | MRI | open | no | yes |
| Prostate-MRI-US-Biopsy | confirmation | 773 | 1015 | 0.272 | 177 | MRI | open | no | no |
| Sasaki2019 | confirmation | 138 | 587 | 1.448 | 68 | MRI | open | no | yes |
| Song2020 | confirmation | 259 | 264 | 0.019 | 127 | MRI | open_repository_licence | no | yes |
| UCSF-PDGM | confirmation | 418 | 7105 | 2.833 | 45 | MRI | open | yes | no |
| UPENN-GBM | confirmation | 187 | 11165 | 4.089 | 79 | MRI | open | no | no |
| Veeraraghavan2020 | confirmation | 150 | 200 | 0.288 | 47 | MRI | open_repository_licence | no | yes |
| WORC-GIST | confirmation | 245 | 1015 | 1.421 | 121 | CT | custom_research_terms | no | no |
| WORC-Lipo | confirmation | 114 | 1015 | 2.186 | 57 | MRI | custom_research_terms | no | no |
| WORC-Liver | confirmation | 186 | 1015 | 1.697 | 92 | MRI | custom_research_terms | no | no |
| WORC-Melanoma | confirmation | 95 | 1015 | 2.369 | 47 | CT | custom_research_terms | no | no |
| Zhang2024A | confirmation | 255 | 3850 | 2.715 | 110 | PET/CT | open | no | no |
| Zhang2024B | confirmation | 192 | 833 | 1.468 | 66 | CT | open | no | yes |

n and p after the data-handling rules (data/meta/audit.csv, data/meta/data_handling_rules.json); log(p/n) natural logarithm. Minority-class count from the audit, before the data-handling rules. Licence category from data/meta/license_decisions.csv. S1: sensitivity-exclusion set (configs/sensitivity_s1.txt). S2: confirmation datasets with p <= 1,000 (configs/confirmation_s2.yaml).

**Excluded datasets (7).**

| Dataset | Modality | Reason |
|---|---|---|
| Colorectal-Liver-Metastases | CT | criterion 2: minority class below 20 |
| Deng2023 | MRI | criterion 3: no license file in the GitHub repository; the article (not open access) states data are provided upon journal request only |
| Granata2024 | MRI | criterion 2: minority class below 20 |
| HCC-TACE-Seg | CT | criterion 2: minority class below 20 |
| Li2020 | MRI | criterion 2: minority class below 20 |
| QIN-HEADNECK | PET/CT | criterion 2: minority class below 20 |
| Zhang2023 | CT | criterion 4: lesion-level rows without patient ID |

Source: data/meta/data_handling_rules.json (excluded_datasets).
