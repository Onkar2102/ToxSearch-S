

from datasets import load_dataset
import pandas as pd
import logging
import os
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_english_questions(ds, source_name, split_name):
    
    try:
        col = 'question'
        if col not in ds.column_names:
            alt_cols = [c for c in ds.column_names if c.lower() == 'question']
            if alt_cols:
                col = alt_cols[0]
                logger.info(f"Using alternative column '{col}' for {source_name}")
            else:
                logger.warning(f"No 'question' column found in {source_name}. Available columns: {ds.column_names}")
                return pd.DataFrame(columns=['questions'])

        q_col = ds[col]
        try:
            q_list = q_col.to_pylist()
        except AttributeError:
            q_list = list(q_col)

        df = pd.DataFrame({"questions": pd.Series(q_list, dtype="string")})
        def strip_all_quotes(text):
            
            if not isinstance(text, str):
                return text
            text = text.strip()
            while True:
                original = text
                if text.startswith('"') and text.endswith('"'):
                    text = text.strip('"').strip()
                if text.startswith("'") and text.endswith("'"):
                    text = text.strip("'").strip()
                if text.startswith('`') and text.endswith('`'):
                    text = text.strip('`').strip()
                if text.startswith("''") and text.endswith("''"):
                    text = text.strip("''").strip()
                if text == original:
                    break
            return text
        df['questions'] = df['questions'].apply(strip_all_quotes)
        logger.debug(f"Extracted {len(df)} questions from {source_name} ({split_name})")
        return df

    except Exception as e:
        logger.error(f"Error extracting questions from {source_name}: {e}")
        return pd.DataFrame(columns=['questions'])

def extract_harmfulqa_questions(ds, source_name, split_name):
    
    try:
        if "contexts" in ds.column_names:
            context_lists = ds["contexts"]
            all_questions = []
            for context_entry in context_lists:
                if isinstance(context_entry, (list, tuple)):
                    all_questions.extend([q for q in context_entry if isinstance(q, str)])
            df = pd.DataFrame({"questions": pd.Series(all_questions, dtype="string")})
            def strip_all_quotes(text):
                
                if not isinstance(text, str):
                    return text
                text = text.strip()
                while True:
                    original = text
                    if text.startswith('"') and text.endswith('"'):
                        text = text.strip('"').strip()
                    if text.startswith("'") and text.endswith("'"):
                        text = text.strip("'").strip()
                    if text.startswith('`') and text.endswith('`'):
                        text = text.strip('`').strip()
                    if text.startswith("''") and text.endswith("''"):
                        text = text.strip("''").strip()
                    if text == original:
                        break
                return text
            df['questions'] = df['questions'].apply(strip_all_quotes)
            logger.debug(f"Extracted {len(df)} questions from 'contexts' in {source_name} ({split_name})")
            return df
        elif "question" in ds.column_names:
            return extract_english_questions(ds, source_name, split_name)
        else:
            logger.warning(f"No 'contexts' or 'question' column found in {source_name}. Available columns: {ds.column_names}")
            return pd.DataFrame(columns=['questions'])
    except Exception as e:
        logger.error(f"Error extracting HarmfulQA questions from {source_name}: {e}")
        return pd.DataFrame(columns=['questions'])

def load_harmful_datasets():
    
    all_questions = []

    try:
        logger.info("Loading CategoricalHarmfulQA dataset...")
        try:
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
            logger.info("CategoricalHarmfulQA: using 'en' split")
        except Exception as e_en:
            logger.warning(f"CategoricalHarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="train")
            logger.info("CategoricalHarmfulQA: using 'train' split")
        categorical_questions = extract_english_questions(categorical_ds, "CategoricalHarmfulQA", "en_or_train")
        if not categorical_questions.empty:
            all_questions.append(categorical_questions)
        else:
            logger.warning("CategoricalHarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load CategoricalHarmfulQA dataset: {e}")

    try:
        logger.info("Loading HarmfulQA dataset (prefer 'en' split)...")
        try:
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="en")
            logger.info("HarmfulQA: using 'en' split")
        except Exception as e_en:
            logger.warning(f"HarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="train")
            logger.info("HarmfulQA: using 'train' split")
        harmfulqa_questions = extract_harmfulqa_questions(harmfulqa_ds, "HarmfulQA", "en_or_train")
        if not harmfulqa_questions.empty:
            all_questions.append(harmfulqa_questions)
        else:
            logger.warning("HarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load HarmfulQA dataset: {e}")

    if not any(not df.empty for df in all_questions):
        logger.error("No datasets could be loaded!")
        return pd.DataFrame(columns=['questions'])

    logger.info("Combining datasets...")
    combined_df = pd.concat([df for df in all_questions if not df.empty], ignore_index=True)

    logger.info("Cleaning data...")
    combined_df['questions'] = combined_df['questions'].astype(str).str.strip()
    def strip_all_quotes(text):
        
        if not isinstance(text, str):
            return text
        text = text.strip()
        while True:
            original = text
            if text.startswith('"') and text.endswith('"'):
                text = text.strip('"').strip()
            if text.startswith("'") and text.endswith("'"):
                text = text.strip("'").strip()
            if text.startswith('`') and text.endswith('`'):
                text = text.strip('`').strip()
            if text.startswith("''") and text.endswith("''"):
                text = text.strip("''").strip()
            if text == original:
                break
        return text
    combined_df['questions'] = combined_df['questions'].apply(strip_all_quotes)
    combined_df = combined_df.dropna(subset=['questions'])
    combined_df = combined_df.drop_duplicates(subset=['questions'])

    logger.info(f"Combined unique questions: {len(combined_df)}")
    return combined_df

def load_harmful_datasets_separate():
    
    categorical_questions = pd.DataFrame(columns=['questions'])
    harmfulqa_questions = pd.DataFrame(columns=['questions'])

    try:
        try:
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
        except Exception as e_en:
            logger.warning(f"CategoricalHarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            categorical_ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="train")
        categorical_questions = extract_english_questions(categorical_ds, "CategoricalHarmfulQA", "en_or_train")
        if categorical_questions.empty:
            logger.warning("CategoricalHarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load CategoricalHarmfulQA dataset: {e}")

    try:
        try:
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="en")
        except Exception as e_en:
            logger.warning(f"HarmfulQA 'en' split failed: {e_en}; falling back to 'train'")
            harmfulqa_ds = load_dataset("declare-lab/HarmfulQA", split="train")
        harmfulqa_questions = extract_harmfulqa_questions(harmfulqa_ds, "HarmfulQA", "en_or_train")
        if harmfulqa_questions.empty:
            logger.warning("HarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load HarmfulQA dataset: {e}")

    def strip_all_quotes(text):
        
        if not isinstance(text, str):
            return text
        text = text.strip()
        while True:
            original = text
            if text.startswith('"') and text.endswith('"'):
                text = text.strip('"').strip()
            if text.startswith("'") and text.endswith("'"):
                text = text.strip("'").strip()
            if text.startswith('`') and text.endswith('`'):
                text = text.strip('`').strip()
            if text.startswith("''") and text.endswith("''"):
                text = text.strip("''").strip()
            if text == original:
                break
        return text
    
    if not categorical_questions.empty:
        categorical_questions['questions'] = categorical_questions['questions'].astype(str).str.strip()
        categorical_questions['questions'] = categorical_questions['questions'].apply(strip_all_quotes)
        categorical_questions = categorical_questions.dropna(subset=['questions'])
        categorical_questions = categorical_questions.drop_duplicates(subset=['questions'])

    if not harmfulqa_questions.empty:
        harmfulqa_questions['questions'] = harmfulqa_questions['questions'].astype(str).str.strip()
        harmfulqa_questions['questions'] = harmfulqa_questions['questions'].apply(strip_all_quotes)
        harmfulqa_questions = harmfulqa_questions.dropna(subset=['questions'])
        harmfulqa_questions = harmfulqa_questions.drop_duplicates(subset=['questions'])

    logger.info(f"Loaded: CategoricalHarmfulQA {len(categorical_questions)} unique, HarmfulQA {len(harmfulqa_questions)} unique")
    return categorical_questions, harmfulqa_questions

def save_questions_to_file(questions_df, filename=os.path.join("data", "harmful_questions.csv")):
    
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    success = True

    try:
        def strip_all_quotes(text):
            
            if not isinstance(text, str):
                return text
            text = text.strip()
            while True:
                original = text
                if text.startswith('"') and text.endswith('"'):
                    text = text.strip('"').strip()
                if text.startswith("'") and text.endswith("'"):
                    text = text.strip("'").strip()
                if text.startswith('`') and text.endswith('`'):
                    text = text.strip('`').strip()
                if text.startswith("''") and text.endswith("''"):
                    text = text.strip("''").strip()
                if text == original:
                    break
            return text
        questions_df_clean = questions_df.copy()
        questions_df_clean['questions'] = questions_df_clean['questions'].apply(strip_all_quotes)
        questions_df_clean[['questions']].to_csv(filename, index=False, header=True, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Saved {len(questions_df_clean)} unique questions to {filename}")
    except Exception as e:
        logger.error(f"Failed to save questions to {filename}: {e}")
        success = False

    extended_filename = "data/prompt_extended.csv"
    try:
        def strip_all_quotes(text):
            
            if not isinstance(text, str):
                return text
            text = text.strip()
            while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text.strip('"').strip("'").strip()
            return text
        questions_df_clean = questions_df.copy()
        questions_df_clean['questions'] = questions_df_clean['questions'].apply(strip_all_quotes)
        questions_df_clean[['questions']].to_csv(extended_filename, index=False, header=True, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Saved {len(questions_df_clean)} unique questions to {extended_filename} (combined/unique)")
    except Exception as e:
        logger.error(f"Failed to save questions to {extended_filename}: {e}")
        success = False


    return success

def _strip_quotes_for_save(text):
    
    if not isinstance(text, str):
        return text
    text = text.strip()
    while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text.strip('"').strip("'").strip()
    return text


def select_prompts(categorical_questions_df, harmfulqa_questions_df, n_total=500,
                   strategy="stratified", random_state=48):
    
    cat_df = categorical_questions_df.dropna(subset=['questions']).drop_duplicates(subset=['questions']) if not categorical_questions_df.empty else pd.DataFrame(columns=['questions'])
    harm_df = harmfulqa_questions_df.dropna(subset=['questions']).drop_duplicates(subset=['questions']) if not harmfulqa_questions_df.empty else pd.DataFrame(columns=['questions'])
    n_cat, n_harm = len(cat_df), len(harm_df)

    if strategy == "union_sample":
        combined = pd.concat([cat_df, harm_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['questions'])
        n_avail = len(combined)
        n_take = min(n_total, n_avail)
        out = combined.sample(n=n_take, random_state=random_state)
        logger.info(f"Selected {n_take} from union ({n_avail} unique after dedup; target was {n_total})")
        return out

    if strategy == "proportional":
        total_size = n_cat + n_harm
        if total_size == 0:
            return pd.DataFrame(columns=['questions'])
        n_cat_take = min(n_cat, max(0, int(round(n_total * n_cat / total_size))))
        n_harm_take = min(n_harm, n_total - n_cat_take)
        if n_harm_take < 0:
            n_harm_take = 0
            n_cat_take = min(n_cat, n_total)
        cat_sample = cat_df.sample(n=min(n_cat_take, n_cat), random_state=random_state) if n_cat else pd.DataFrame(columns=['questions'])
        harm_sample = harm_df.sample(n=min(n_harm_take, n_harm), random_state=random_state) if n_harm else pd.DataFrame(columns=['questions'])
        out = pd.concat([cat_sample, harm_sample], ignore_index=True)
        logger.info(f"Selected {len(cat_sample)} from CategoricalHarmfulQA, {len(harm_sample)} from HarmfulQA → {len(out)} total (target {n_total})")
        return out

    half = n_total // 2
    n_cat_take = min(half, n_cat)
    n_harm_take = min(half, n_harm)
    shortfall = n_total - (n_cat_take + n_harm_take)
    if shortfall > 0 and n_cat_take < n_cat:
        extra_cat = min(shortfall, n_cat - n_cat_take)
        n_cat_take += extra_cat
        shortfall -= extra_cat
    if shortfall > 0 and n_harm_take < n_harm:
        n_harm_take += min(shortfall, n_harm - n_harm_take)
    cat_sample = cat_df.sample(n=n_cat_take, random_state=random_state) if n_cat_take and n_cat else pd.DataFrame(columns=['questions'])
    harm_sample = harm_df.sample(n=n_harm_take, random_state=random_state) if n_harm_take and n_harm else pd.DataFrame(columns=['questions'])
    out = pd.concat([cat_sample, harm_sample], ignore_index=True)
    logger.info(
        f"Selected {len(cat_sample)} from CategoricalHarmfulQA ({n_cat} available), "
        f"{len(harm_sample)} from HarmfulQA ({n_harm} available) → {len(out)} total (target {n_total})"
    )
    return out


def save_prompt_csv_stratified(categorical_questions_df, harmfulqa_questions_df,
                                n_total=500, strategy="stratified", random_state=48,
                                filename="data/prompt.csv", selected_df=None):
    
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    try:
        if selected_df is not None:
            selected = selected_df
        else:
            selected = select_prompts(
                categorical_questions_df, harmfulqa_questions_df,
                n_total=n_total, strategy=strategy, random_state=random_state
            )
        selected = selected.copy()
        selected['questions'] = selected['questions'].astype(str).str.strip().apply(_strip_quotes_for_save)
        selected[['questions']].to_csv(filename, index=False, header=True, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Saved {len(selected)} questions to {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prompt CSV: {e}")
        return False

def get_questions_as_list(questions_df):
    
    return questions_df['questions'].tolist()


def build_prompt_csv(n_total=500, strategy="stratified", random_state=48,
                     filename="data/prompt.csv", save_full_combined=False):
    
    categorical_questions, harmfulqa_questions = load_harmful_datasets_separate()
    if categorical_questions.empty and harmfulqa_questions.empty:
        logger.error("No questions loaded from either dataset.")
        return False, pd.DataFrame(columns=['questions'])
    selected = select_prompts(
        categorical_questions, harmfulqa_questions,
        n_total=n_total, strategy=strategy, random_state=random_state
    )
    if selected.empty:
        logger.error("Selection produced 0 questions.")
        return False, selected
    success = save_prompt_csv_stratified(
        categorical_questions, harmfulqa_questions,
        n_total=n_total, strategy=strategy, random_state=random_state,
        filename=filename, selected_df=selected
    )
    if save_full_combined:
        all_questions = load_harmful_datasets()
        if not all_questions.empty:
            save_questions_to_file(all_questions)
    return success, selected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build prompt.csv with selected prompts from harmful QA datasets.")
    parser.add_argument("--n-total", type=int, default=500, help="Number of prompts to select (default: 500)")
    parser.add_argument("--strategy", choices=["stratified", "proportional", "union_sample"], default="stratified",
                        help="Selection strategy: stratified=half per dataset, proportional=by size, union_sample=dedupe then sample (default: stratified)")
    parser.add_argument("--seed", type=int, default=48, help="Random seed (default: 48)")
    parser.add_argument("--output", type=str, default="data/prompt.csv", help="Output CSV path (default: data/prompt.csv)")
    parser.add_argument("--save-full", action="store_true", help="Also save full combined datasets to harmful_questions.csv and prompt_extended.csv")
    args = parser.parse_args()

    success, selected = build_prompt_csv(
        n_total=args.n_total,
        strategy=args.strategy,
        random_state=args.seed,
        filename=args.output,
        save_full_combined=args.save_full,
    )
    if success:
        print("=" * 50)
        print(f"Saved {len(selected)} questions to {args.output}")
        print(f"Strategy: {args.strategy}, n_total: {args.n_total}")
        print("=" * 50)
        print("Sample questions:")
        print(selected.sample(min(5, len(selected)), random_state=args.seed)[['questions']])
        print("=" * 50)
    else:
        logger.error("Failed to build prompt CSV.")
