from src.analysis.evaluation_model import full_model_evaluation, write_results_to_excel
from src.parameters import model_parameters as p




model_results, comparison_table = full_model_evaluation(9.903657, 1.0, 0.841135, 1.399776, 400.0, 2.02879, p)

write_results_to_excel(
    results_dict=model_results,
    comparison_list=comparison_table,
    mode="TOC"
)

