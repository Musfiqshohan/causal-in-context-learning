from .eval_function import eval_fun_sentiment, eval_fun_llm_ret_nli, eval_fun_coreference, eval_fun_commonsense
from .eval_function import eval_fun_sa, eval_fun_boss_nli, eval_fun_td

class EvalLLM:
    def __init__(self, benchmark='ood_nlp', task='nli', domain='mnli', llm='llama', gt_answers=None):
        self.benchmark = benchmark
        self.domain = domain
        self.llm = llm
        self.gt_answers = gt_answers

        if domain in ['mnli', 'anli', 'contract_nli', 'wanli']:
            self.eval = eval_fun_boss_nli.EvalFun()
        elif domain in ['qnli', 'rte', 'wsc']:
            self.eval = eval_fun_llm_ret_nli.EvalFun()
        else: self.eval = eval(f"eval_fun_{task}").EvalFun()

        if benchmark == 'llm_ret':
            if llm in ['qwen306b', 'llama']:
                self.marker = "</Answer>assistant"
            elif llm in ['bitnet']:
                self.marker = "</Answer>Assistant:"
            elif llm in ['phi4-mini-it']:
                self.marker = "</Answer>"
        elif benchmark == 'ood_nlp': self.marker = "Prediction"

    def __call__(self, response_list, subset_ids):
        return self.eval(response_list, self.gt_answers, subset_ids, marker=self.marker)