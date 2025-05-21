import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import scipy.stats as stats
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm

class SurveyAnalysisAgent:
    """
    Agent to analyze survey data: automatically select appropriate significance test
    based on classical decision toolkit and generate an academic results paragraph.
    """
    def __init__(self, reference_file='reference.xlsx', api_key=None, temperature=0.7):
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        self.client = OpenAI(api_key=self.api_key)
        self.temperature = temperature
        self.references = self._load_references(reference_file)

    def _load_references(self, reference_path):
        df = pd.read_excel(reference_path)
        examples = []
        for _, row in df.iterrows():
            examples.append({
                'survey_design': str(row['survey design']),
                'dataset_example': row['dataset(example)'],
                'result_paragraph': str(row['result paragraph'])
            })
        return examples

    def _format_dataset_preview(self, df):
        return df.head(10).to_csv(index=False)

    def _is_categorical(self, series: pd.Series) -> bool:
        return (
            pd.api.types.is_categorical_dtype(series)
            or series.dtype == 'object'
            or (pd.api.types.is_integer_dtype(series) and series.nunique() == 2)
        )

    def _choose_test(self, df, dv_col, iv_col=None, paired=False, known_pop=False):
        # DEBUG: print DV/IV and types
        # print(f"[DEBUG] DV column: {dv_col}, IV column: {iv_col}")
        # print(f"[DEBUG] DV dtype: {df[dv_col].dtype}, unique values: {df[dv_col].unique()[:5]}")
        # if iv_col:
        #     print(f"[DEBUG] IV dtype: {df[iv_col].dtype}, unique values: {df[iv_col].unique()[:5]}")

        # Categorical DV?
        if self._is_categorical(df[dv_col]):
            if iv_col:
                return 'Chi-Square Test'
            else:
                return 'Descriptive Statistics'

        # Numeric DV
        if paired and iv_col:
            return 'Paired samples t-test'

        # Two-group numeric comparison
        if iv_col and df[iv_col].nunique() == 2:
            if known_pop:
                return 'Z-test'
            groups = df[iv_col].dropna().unique()
            vtest = stats.levene(*(df[df[iv_col]==g][dv_col].dropna() for g in groups))
            # print(f"[DEBUG] Levene test p-value: {vtest.pvalue}")
            if vtest.pvalue < 0.05:
                return 'Welch t-test'
            else:
                return 'Independent samples t-test'

        # Multi-group numeric
        if iv_col and df[iv_col].nunique() > 2:
            return 'ANOVA'

        # Nonparametric fallback
        if iv_col and df[iv_col].nunique() == 2:
            return 'Mann-Whitney U test'
        if iv_col and df[iv_col].nunique() > 2:
            return 'Kruskal-Wallis test'

        # Distribution comparison
        if iv_col and df[iv_col].nunique() == 2:
            return 'Kolmogorov-Smirnov test'

        return 'Descriptive Statistics'

    def _ask_gpt_for_analysis_plan(self, survey_design_text: str, dataset_preview: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a data scientist. Based on the survey design and dataset preview, "
                    "propose an analysis plan in valid JSON with 'analysis_plan' list of objects, "
                    "each having 'statistical_method' and 'variables'."
                )},
                {"role": "user", "content": (
                    f"Survey Design:\n{survey_design_text}\n\nDataset Preview:\n{dataset_preview}\n"
                    "Please output only the JSON structure."
                )}
            ],
            temperature=self.temperature
        )
        import json
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)

    def _run_real_analysis(self, df, plan: dict):
        # Determine DV and IV from plan
        dv_col = plan['analysis_plan'][0]['variables'][0]
        iv_col = plan['analysis_plan'][0]['variables'][1] if len(plan['analysis_plan'][0]['variables']) > 1 else None
        test = self._choose_test(df, dv_col, iv_col)
        # Chi-Square Test
        if test == 'Chi-Square Test':
            ct = pd.crosstab(df[dv_col], df[iv_col])
            n = ct.values.sum()
            myst_idx = 1 if 1 in ct.index else ct.index[0]
            myst_count = ct.loc[myst_idx].sum()
            myst_pct = myst_count / n * 100
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            phi = (chi2 / n / (min(ct.shape) - 1)) ** 0.5
            result_str = (
                f"n={n}, {myst_count}/{n} ({myst_pct:.1f}%) chose the mystery option; "
                f"χ²({dof})={chi2:.2f}, p={p:.3f}, φ={phi:.2f}"
            )
            return test, result_str
        # Independent Samples T-Test
        if test == 'Independent samples t-test':
            groups = df[iv_col].unique()
            g1 = df[df[iv_col] == groups[0]][dv_col].dropna()
            g2 = df[df[iv_col] == groups[1]][dv_col].dropna()
            t_stat, p = stats.ttest_ind(g1, g2, nan_policy='omit')
            n1, n2 = len(g1), len(g2)
            m1, m2 = g1.mean(), g2.mean()
            sd1, sd2 = g1.std(), g2.std()
            pooled = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2)
            pooled_sd = pooled ** 0.5
            d = (m1 - m2) / pooled_sd if pooled_sd else float('nan')
            se = pooled_sd * (1 / n1 + 1 / n2) ** 0.5
            df_t = n1 + n2 - 2
            crit = stats.t.ppf(0.975, df_t)
            ci_low, ci_up = (m1 - m2) - crit * se, (m1 - m2) + crit * se
            result_str = (
                f"M1={m1:.2f}, SD1={sd1:.2f}, M2={m2:.2f}, SD2={sd2:.2f}; "
                f"t({df_t})={t_stat:.2f}, p={p:.3f}, d={d:.2f}, 95% CI [{ci_low:.2f}, {ci_up:.2f}]"
            )
            return test, result_str
        # Welch's T-Test
        if test == 'Welch t-test':
            groups = df[iv_col].unique()
            g1 = df[df[iv_col] == groups[0]][dv_col].dropna()
            g2 = df[df[iv_col] == groups[1]][dv_col].dropna()
            t_stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
            result_str = f"t={t_stat:.2f}, p={p:.3f}"
            return test, result_str
        # Paired Samples T-Test
        if test == 'Paired samples t-test':
            groups = df[iv_col].unique()
            a = df[df[iv_col] == groups[0]][dv_col].dropna()
            b = df[df[iv_col] == groups[1]][dv_col].dropna()
            t_stat, p = stats.ttest_rel(a, b)
            result_str = f"t={t_stat:.2f}, p={p:.3f}"
            return test, result_str
        # ANOVA
        if test == 'ANOVA':
            groups = [df[df[iv_col] == lvl][dv_col].dropna() for lvl in df[iv_col].unique()]
            f_stat, p = stats.f_oneway(*groups)
            result_str = f"F={f_stat:.2f}, p={p:.3f}"
            return test, result_str
        # Mann-Whitney U Test
        if test == 'Mann-Whitney U test':
            groups = df[iv_col].unique()
            u_stat, p = stats.mannwhitneyu(
                df[df[iv_col] == groups[0]][dv_col].dropna(),
                df[df[iv_col] == groups[1]][dv_col].dropna(),
                alternative='two-sided'
            )
            result_str = f"U={u_stat:.2f}, p={p:.3f}"
            return test, result_str
        # Kruskal-Wallis Test
        if test == 'Kruskal-Wallis test':
            groups = [df[df[iv_col] == lvl][dv_col].dropna() for lvl in df[iv_col].unique()]
            h_stat, p = stats.kruskal(*groups)
            result_str = f"H={h_stat:.2f}, p={p:.3f}"
            return test, result_str
        # Kolmogorov-Smirnov Test
        if test == 'Kolmogorov-Smirnov test':
            groups = df[iv_col].unique()
            ks_stat, p = stats.ks_2samp(
                df[df[iv_col] == groups[0]][dv_col].dropna(),
                df[df[iv_col] == groups[1]][dv_col].dropna()
            )
            result_str = f"KS={ks_stat:.2f}, p={p:.3f}"
            return test, result_str
        # Fallback descriptive
        if test == 'Descriptive Statistics':
            desc = df[[dv_col]].describe().T[['mean','std','count']]
            m = desc.loc[dv_col,'mean']
            sd = desc.loc[dv_col,'std']
            n = int(desc.loc[dv_col,'count'])
            result_str = f"M={m:.2f}, SD={sd:.2f}, N={n}"
            return test, result_str
        return test, ''


    def _build_prompt_with_results(self, survey_design_text, preview, test, result_txt):
        parts = []
        for ex in self.references:
            preview_ex = ex['dataset_example'] if not isinstance(ex['dataset_example'], pd.DataFrame) else ex['dataset_example'].head(5).to_csv(index=False)
            parts.append(
                f"### Example Survey Design\n{ex['survey_design']}\n"
                f"### Dataset Preview\n{preview_ex}\n"
                f"### Example Analysis Paragraph\n{ex['result_paragraph']}\n"
            )
        instruction = (
            "### Task Instruction:\n"
            "- Based strictly on the provided descriptive and inferential statistics, write an academic results paragraph.\n"
            "- Report observed proportions, test statistics, effect sizes, and confidence intervals accurately.\n"
            "- Do not invent or fabricate any data; omit statements for which no statistics are provided.\n"
            "- Avoid referencing hypothesis labels; conclude whether results support expectations.\n"
            "- Include limitations regarding sample size, design constraints, and suggestions for future work."
        )
        parts.append(
            f"### New Survey Design\n{survey_design_text}\n"
            f"### Dataset Preview\n{preview}\n"
            f"### Selected Test: {test}\n"
            f"### Statistical Output\n{result_txt}\n"
            + instruction
        )
        return "\n---\n".join(parts)

    def process(self, survey_design_path, dataset_path):
        survey_design = Path(survey_design_path).read_text(encoding='utf-8')
        df = pd.read_csv(dataset_path)
        preview = self._format_dataset_preview(df)
        plan = self._ask_gpt_for_analysis_plan(survey_design, preview)
        test, result_txt = self._run_real_analysis(df, plan)
        prompt = self._build_prompt_with_results(survey_design, preview, test, result_txt)
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a PhD-level survey methods expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_file")
    parser.add_argument("survey_design")
    parser.add_argument("dataset")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    agent = SurveyAnalysisAgent(reference_file=args.reference_file, temperature=args.temperature)
    print(agent.process(args.survey_design, args.dataset))

