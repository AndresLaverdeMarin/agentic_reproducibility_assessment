models = {
    "gemini-2.5-flash"
    ? # 4 models maybe, available via Gemini API

    # also maybe locally GPU from Andres some OpenSource
    
}

temperatures = {
    0, 1, 2, 3, 4, 5 # how many and which temperatures?
}

papers_to_check = { # selected based on their length
    2017_04_article.pdf # pages 4   3372
    2020_26_article.pdf # pages 8   3701
    2017_01_article.pdf # pages 15  6436
    2025_03_article.pdf # pages 20	8930
    2017_08_article.pdf # pages 11  9237
    2022_30_article.pdf # pages 18	9800
    2021_34_article.pdf # pages 28	13493
    2023_17_article.pdf # pages 24	14733
    2023_15_article.pdf # pages 30	16170
    2022_38_article.pdf # pages 34	22917
}

N TIMES

def ara_pipeline_function(paper_pdf_path):
    # placeholder function
    graph = ara_graph_builder(document=paper_pdf_path)
    micro_scores = ara_micro_scoring(document=paper_pdf_path, graph)
    macro_scores = ara_macro_scoring(document=paper_pdf_path, graph, micro_scores)
    store to json file the run in "data/experiments/consistency_checks"
    like "paperid_temp_model_sample.json"

# Run ARA
for model in models:
    for temperature in temperatures:
        for paper in papers_to_check:
            ara_pipeline_function(paper)

# Analyse Results, create python file in src/_figures_tables/consistency_check.py to actually render tables and figures
# Table 1: consistency across temperatures
# Table 2: consistency across paper lengths
# Table 3: consistency across models