
from neuro_process import run_learn_individual
from neuro_process import run_learn_combined

print("Dataset UPenn")
upenn_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/UPenn_Multiple_Neurodegenerative_Diseases/Discovery_LFQ_Proteomics/data/0.Traits.csv"
upenn_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/UPenn_Multiple_Neurodegenerative_Diseases/Discovery_LFQ_Proteomics/data/2.unregressed_batch-corrected_LFQ_intensity.csv"
upenn_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/upenn_joint.xlsx"
run_learn_individual(upenn_proteomics_file, 'Unnamed: 0', upenn_traits_file, 'MaxQuant ID', upenn_excel_file, "Group")

# print("Dataset Mayo")
# mayo_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Mayo_Temporal_Cortex/data/0.Traits.csv"
# mayo_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Mayo_Temporal_Cortex/data/2.unregressed_batch-corrected_LFQ_intensity.csv"
# mayo_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/mayo_joint.xlsx"
# run_learn_individual(mayo_proteomics_file, 'Unnamed: 0', mayo_traits_file, 'SampleID', mayo_excel_file, "Diagnosis")
#
# print("Dataset CSF 298")
# csf_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_1_298-sample_FNIH_Symptomatic_AD_and_Controls/data/0.Traits.csv"
# csf_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_1_298-sample_FNIH_Symptomatic_AD_and_Controls/data/2.Unregressed_Batch-corrected_PD-normalized_TMT_reporter_abundance.csv"
# csf_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/csf_298_joint.xlsx"
# run_learn_individual(csf_proteomics_file, 'Unnamed: 0', csf_traits_file, 'SampleID', csf_excel_file, "Group")
#
# print("Dataset CSF 298 clean")
# csf_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_1_298-sample_FNIH_Symptomatic_AD_and_Controls/data/0.Traits.csv"
# csf_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_1_298-sample_FNIH_Symptomatic_AD_and_Controls/data/2b.Unregressed_Batch-corrected_cleanDat.csv"
# csf_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/csf_298_joint_clean.xlsx"
# run_learn_individual(csf_proteomics_file, 'Unnamed: 0', csf_traits_file, 'SampleID', csf_excel_file, "Group")
#
# print("Dataset CSF 96")
# csf_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_2_96-sample_biomarker_defined_Asymptomatic_AD_Symptomatic_AD_and_Controls/data/0.Traits.csv"
# csf_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_2_96-sample_biomarker_defined_Asymptomatic_AD_Symptomatic_AD_and_Controls/data/2.Unregressed_Batch-corrected_PD-TMT_reporter_abundance.csv"
# csf_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/csf_96_joint.xlsx"
# run_learn_individual(csf_proteomics_file, 'Unnamed: 0', csf_traits_file, 'batch.channel', csf_excel_file, "ClinicalGroup")
#
# print("Dataset CSF 96 clean")
# csf_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_2_96-sample_biomarker_defined_Asymptomatic_AD_Symptomatic_AD_and_Controls/data/0.Traits.csv"
# csf_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/CSF/Cohort_2_96-sample_biomarker_defined_Asymptomatic_AD_Symptomatic_AD_and_Controls/data/2b.unregressed_Batch-corrected_cleanDat.csv"
# csf_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/csf_96_joint_clean.xlsx"
# run_learn_individual(csf_proteomics_file, 'Unnamed: 0', csf_traits_file, 'batch.channel', csf_excel_file, "ClinicalGroup")
#
# #Age, regression
# # aging_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Aging/data/0.Traits.csv"
# # aging_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Aging/data/2b.unregressed_Batch-corrected_cleanDat.csv"
# # aging_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/aging_joint.xlsx"
# # run_learn(aging_proteomics_file, 'Unnamed: 0', aging_traits_file, 'batch.channel', aging_excel_file, "ClinicalGroup")
#
# print("Dataset Consensus")
# consensus_traits_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Consensus/data/0.Traits.csv"
# consensus_proteomics_file = "/Users/frishman/Dropbox/Deeproad/data/johnson_20/Consensus/data/2.Minimally_regressed_Batch_and_Site-corrected_LFQ_intensity.csv"
# consensus_excel_file = "/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/consensus_joint.xlsx"
# run_learn_individual(consensus_proteomics_file, 'Unnamed: 0', consensus_traits_file, 'Unnamed: 0', consensus_excel_file, "Group")
#
# print("Dataset combat")
# run_learn_combined('/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/adata_combined_integrated_nor1_agewithoutgroup_sex_pmi_state_combat_del_CBD_ALSFTD.h5ad')
#
# print("Dataset desc")
# run_learn_combined('/Users/frishman/Dropbox/Bioinformatics/projects/Neuro/adata_combined_integrated_nor1_agewithoutgroup_sex_pmi_state_desc_del_CBD_ALSFTD.h5ad')
#


