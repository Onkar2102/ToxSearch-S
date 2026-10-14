

from datasets import load_dataset
import pandas as pd
import logging
import os
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_english_questions(ds, source_name, split_name, column="question"):
    """Extract a text column from a HF Dataset into a DataFrame with ['questions']."""
    try:
        col = column
        if col not in ds.column_names:
            alt_cols = [c for c in ds.column_names if c.lower() == column.lower()]
            if alt_cols:
                col = alt_cols[0]
                logger.info(f"Using alternative column '{col}' for {source_name}")
            else:
                logger.warning(
                    f"No '{column}' column found in {source_name}. Available columns: {ds.column_names}"
                )
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
        logger.debug(f"Extracted {len(df)} questions from {source_name} ({split_name}) column '{col}'")
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


def extract_do_not_answer_questions(ds, source_name="DoNotAnswer", split_name="train",
                                    require_endswith_qmark=True):
    """Extract `question` from LibrAI/do-not-answer; keep only strings ending with '?' by default."""
    try:
        df = extract_english_questions(ds, source_name, split_name)
        if df.empty:
            return df
        df = df.copy()
        df["questions"] = df["questions"].astype(str).str.strip()
        before = len(df)
        if require_endswith_qmark:
            df = df[df["questions"].str.endswith("?")].copy()
            logger.info(
                "DoNotAnswer: kept %d/%d questions ending with '?' (%s / %s)",
                len(df), before, source_name, split_name,
            )
        df = df.dropna(subset=["questions"]).drop_duplicates(subset=["questions"]).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error extracting DoNotAnswer questions from {source_name}: {e}")
        return pd.DataFrame(columns=["questions"])


def _clean_questions_frame(df, require_endswith_qmark=True, source_name=None):
    """Strip whitespace/quotes, optionally keep only prompts ending with '?', then dedupe."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["questions"])

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
            if text.startswith("`") and text.endswith("`"):
                text = text.strip("`").strip()
            if text.startswith("''") and text.endswith("''"):
                text = text.strip("''").strip()
            if text == original:
                break
        return text

    out = df.copy()
    out["questions"] = out["questions"].astype(str).str.strip().apply(strip_all_quotes)
    out = out.dropna(subset=["questions"])
    before = len(out)
    if require_endswith_qmark:
        out = out[out["questions"].str.endswith("?")].copy()
        dropped = before - len(out)
        if dropped:
            label = source_name or "pool"
            logger.info(
                "%s: dropped %d/%d prompts not ending with '?' before sampling",
                label, dropped, before,
            )
    out = out.drop_duplicates(subset=["questions"]).reset_index(drop=True)
    return out


def _sample_ending_with_qmark(src_df, k, random_state, source_name):
    """Sample k prompts that end with '?'; resample replacements if any invalid slip through."""
    empty = pd.DataFrame(columns=["questions"])
    if k <= 0 or src_df is None or src_df.empty:
        return empty

    pool = _clean_questions_frame(src_df, require_endswith_qmark=True, source_name=source_name)
    if pool.empty:
        logger.warning("%s: no prompts ending with '?' available to sample", source_name)
        return empty

    n_take = min(k, len(pool))
    if n_take < k:
        logger.warning(
            "%s: requested %d but only %d prompts ending with '?' available",
            source_name, k, n_take,
        )

    rng = random_state
    selected = pool.sample(n=n_take, random_state=rng).copy()
    # Defensive: replace any non-'?' rows by resampling from the remaining pool.
    max_rounds = 20
    for round_i in range(max_rounds):
        bad_mask = ~selected["questions"].astype(str).str.endswith("?")
        n_bad = int(bad_mask.sum())
        if n_bad == 0:
            break
        logger.warning(
            "%s: %d sampled prompt(s) did not end with '?'; resampling (round %d)",
            source_name, n_bad, round_i + 1,
        )
        remaining = pool[~pool["questions"].isin(selected.loc[~bad_mask, "questions"])]
        remaining = _clean_questions_frame(remaining, require_endswith_qmark=True, source_name=source_name)
        if remaining.empty:
            selected = selected.loc[~bad_mask].copy()
            logger.error(
                "%s: could not replace %d invalid prompt(s); keeping %d valid",
                source_name, n_bad, len(selected),
            )
            break
        rng = (rng + 1) if isinstance(rng, int) else random_state
        replacements = remaining.sample(n=min(n_bad, len(remaining)), random_state=rng)
        selected = pd.concat(
            [selected.loc[~bad_mask], replacements],
            ignore_index=True,
        )

    selected = selected[
        selected["questions"].astype(str).str.endswith("?")
    ].drop_duplicates(subset=["questions"]).reset_index(drop=True)
    return selected



def load_do_not_answer(require_endswith_qmark=True):
    """Load LibrAI/do-not-answer train split; optionally keep only questions ending with '?'."""
    try:
        logger.info("Loading DoNotAnswer dataset (LibrAI/do-not-answer, split=train)...")
        dna_ds = load_dataset("LibrAI/do-not-answer", split="train")
        dna_questions = extract_do_not_answer_questions(
            dna_ds,
            source_name="DoNotAnswer",
            split_name="train",
            require_endswith_qmark=require_endswith_qmark,
        )
        dna_questions = _clean_questions_frame(
            dna_questions, require_endswith_qmark=require_endswith_qmark, source_name="DoNotAnswer"
        )
        if dna_questions.empty:
            logger.warning("DoNotAnswer produced 0 questions after extraction/filter.")
        else:
            logger.info("DoNotAnswer: %d unique questions after filter", len(dna_questions))
        return dna_questions
    except Exception as e:
        logger.error(f"Failed to load DoNotAnswer dataset: {e}")
        return pd.DataFrame(columns=["questions"])


def extract_beavertails_prompts(ds, source_name="BeaverTails", split_name="330k_train",
                                require_endswith_qmark=True):
    """Extract unique `prompt` values from BeaverTails; keep only strings ending with '?' by default."""
    try:
        df = extract_english_questions(ds, source_name, split_name, column="prompt")
        if df.empty:
            return df
        df = df.copy()
        df["questions"] = df["questions"].astype(str).str.strip()
        before_rows = len(df)
        before_unique = df["questions"].nunique()
        if require_endswith_qmark:
            df = df[df["questions"].str.endswith("?")].copy()
            logger.info(
                "BeaverTails: kept %d/%d rows ending with '?' "
                "(%d/%d unique prompts) (%s / %s)",
                len(df), before_rows,
                df["questions"].nunique(), before_unique,
                source_name, split_name,
            )
        df = df.dropna(subset=["questions"]).drop_duplicates(subset=["questions"]).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error extracting BeaverTails prompts from {source_name}: {e}")
        return pd.DataFrame(columns=["questions"])


def load_beavertails(split="330k_train", require_endswith_qmark=True):
    """Load PKU-Alignment/BeaverTails; keep unique prompts ending with '?' by default."""
    try:
        logger.info("Loading BeaverTails dataset (PKU-Alignment/BeaverTails, split=%s)...", split)
        bt_ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
        bt_questions = extract_beavertails_prompts(
            bt_ds,
            source_name="BeaverTails",
            split_name=split,
            require_endswith_qmark=require_endswith_qmark,
        )
        bt_questions = _clean_questions_frame(
            bt_questions, require_endswith_qmark=require_endswith_qmark, source_name="BeaverTails"
        )
        if bt_questions.empty:
            logger.warning("BeaverTails produced 0 questions after extraction/filter.")
        else:
            logger.info("BeaverTails: %d unique questions after filter", len(bt_questions))
        return bt_questions
    except Exception as e:
        logger.error(f"Failed to load BeaverTails dataset: {e}")
        return pd.DataFrame(columns=["questions"])


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

    dna_questions = load_do_not_answer(require_endswith_qmark=True)
    if not dna_questions.empty:
        all_questions.append(dna_questions)

    bt_questions = load_beavertails(split="330k_train", require_endswith_qmark=True)
    if not bt_questions.empty:
        all_questions.append(bt_questions)

    if not any(not df.empty for df in all_questions):
        logger.error("No datasets could be loaded!")
        return pd.DataFrame(columns=['questions'])

    logger.info("Combining datasets...")
    combined_df = pd.concat([df for df in all_questions if not df.empty], ignore_index=True)
    combined_df = _clean_questions_frame(
        combined_df, require_endswith_qmark=True, source_name="combined"
    )

    logger.info(f"Combined unique questions: {len(combined_df)}")
    return combined_df

def load_harmful_datasets_separate():
    
    categorical_questions = pd.DataFrame(columns=['questions'])
    harmfulqa_questions = pd.DataFrame(columns=['questions'])
    do_not_answer_questions = pd.DataFrame(columns=['questions'])
    beavertails_questions = pd.DataFrame(columns=['questions'])

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

    do_not_answer_questions = load_do_not_answer(require_endswith_qmark=True)
    beavertails_questions = load_beavertails(split="330k_train", require_endswith_qmark=True)

    categorical_questions = _clean_questions_frame(
        categorical_questions, require_endswith_qmark=True, source_name="CategoricalHarmfulQA"
    )
    harmfulqa_questions = _clean_questions_frame(
        harmfulqa_questions, require_endswith_qmark=True, source_name="HarmfulQA"
    )
    do_not_answer_questions = _clean_questions_frame(
        do_not_answer_questions, require_endswith_qmark=True, source_name="DoNotAnswer"
    )
    beavertails_questions = _clean_questions_frame(
        beavertails_questions, require_endswith_qmark=True, source_name="BeaverTails"
    )

    logger.info(
        "Loaded: CategoricalHarmfulQA %d unique, HarmfulQA %d unique, "
        "DoNotAnswer %d unique, BeaverTails %d unique",
        len(categorical_questions), len(harmfulqa_questions),
        len(do_not_answer_questions), len(beavertails_questions),
    )
    return categorical_questions, harmfulqa_questions, do_not_answer_questions, beavertails_questions


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


def select_prompts(categorical_questions_df, harmfulqa_questions_df,
                   do_not_answer_df=None, beavertails_df=None, n_total=500,
                   strategy="stratified", random_state=48):
    # Process each source BEFORE sampling: strip/dedupe and keep only prompts ending with '?'.
    cat_df = _clean_questions_frame(categorical_questions_df, require_endswith_qmark=True, source_name="CategoricalHarmfulQA")
    harm_df = _clean_questions_frame(harmfulqa_questions_df, require_endswith_qmark=True, source_name="HarmfulQA")
    dna_df = _clean_questions_frame(do_not_answer_df, require_endswith_qmark=True, source_name="DoNotAnswer")
    bt_df = _clean_questions_frame(beavertails_df, require_endswith_qmark=True, source_name="BeaverTails")
    n_cat, n_harm, n_dna, n_bt = len(cat_df), len(harm_df), len(dna_df), len(bt_df)
    sources = [
        ("CategoricalHarmfulQA", cat_df, n_cat),
        ("HarmfulQA", harm_df, n_harm),
        ("DoNotAnswer", dna_df, n_dna),
        ("BeaverTails", bt_df, n_bt),
    ]
    avail = {name: n for name, _df, n in sources}

    if strategy == "union_sample":
        combined = pd.concat([cat_df, harm_df, dna_df, bt_df], ignore_index=True)
        combined = _clean_questions_frame(combined, require_endswith_qmark=True, source_name="union")
        out = _sample_ending_with_qmark(combined, n_total, random_state, "union")
        logger.info(
            "Selected %d from union (%d unique ending with '?'; target was %d)",
            len(out), len(combined), n_total,
        )
        return out

    if strategy == "proportional":
        total_size = sum(avail.values())
        if total_size == 0:
            return pd.DataFrame(columns=['questions'])
        raw = {name: n_total * n / total_size for name, n in avail.items()}
        takes = {name: min(int(raw[name]), avail[name]) for name in avail}
        shortfall = n_total - sum(takes.values())
        for name, _df, n_avail in sorted(sources, key=lambda x: x[2], reverse=True):
            if shortfall <= 0:
                break
            room = n_avail - takes[name]
            if room <= 0:
                continue
            add = min(shortfall, room)
            takes[name] += add
            shortfall -= add
        samples = []
        for name, src_df, _n in sources:
            samples.append(_sample_ending_with_qmark(src_df, takes[name], random_state, name))
        out = pd.concat(samples, ignore_index=True)
        out = _clean_questions_frame(out, require_endswith_qmark=True, source_name="selected")
        # If short after filtering, top up from remaining union pool.
        if len(out) < n_total:
            remaining = pd.concat(
                [df for _name, df, n in sources if n > 0], ignore_index=True
            )
            remaining = remaining[~remaining["questions"].isin(out["questions"])]
            need = n_total - len(out)
            topup = _sample_ending_with_qmark(remaining, need, random_state + 7, "topup")
            out = pd.concat([out, topup], ignore_index=True)
            out = _clean_questions_frame(out, require_endswith_qmark=True, source_name="selected")
        logger.info(
            "Selected %d CategoricalHarmfulQA, %d HarmfulQA, %d DoNotAnswer, %d BeaverTails → %d total (target %d)",
            len(samples[0]), len(samples[1]), len(samples[2]), len(samples[3]), len(out), n_total,
        )
        return out

    # stratified: equal share across available sources, then fill shortfall
    nonempty = [(name, src_df, n) for name, src_df, n in sources if n > 0]
    if not nonempty:
        return pd.DataFrame(columns=['questions'])
    per = n_total // len(nonempty)
    takes = {name: 0 for name, _, _ in sources}
    for name, _src_df, n_avail in nonempty:
        takes[name] = min(per, n_avail)
    shortfall = n_total - sum(takes.values())
    for name, _src_df, n_avail in nonempty:
        if shortfall <= 0:
            break
        room = n_avail - takes[name]
        if room <= 0:
            continue
        add = min(shortfall, room)
        takes[name] += add
        shortfall -= add

    samples = []
    for name, src_df, _n in sources:
        samples.append(_sample_ending_with_qmark(src_df, takes[name], random_state, name))
    out = pd.concat(samples, ignore_index=True)
    out = _clean_questions_frame(out, require_endswith_qmark=True, source_name="selected")
    if len(out) < n_total:
        remaining = pd.concat([df for _name, df, n in sources if n > 0], ignore_index=True)
        remaining = remaining[~remaining["questions"].isin(out["questions"])]
        need = n_total - len(out)
        topup = _sample_ending_with_qmark(remaining, need, random_state + 11, "topup")
        out = pd.concat([out, topup], ignore_index=True)
        out = _clean_questions_frame(out, require_endswith_qmark=True, source_name="selected")
    logger.info(
        "Selected %d from CategoricalHarmfulQA (%d available), "
        "%d from HarmfulQA (%d available), "
        "%d from DoNotAnswer (%d available), "
        "%d from BeaverTails (%d available) → %d total (target %d); "
        "all end with '?': %s",
        len(samples[0]), n_cat, len(samples[1]), n_harm,
        len(samples[2]), n_dna, len(samples[3]), n_bt, len(out), n_total,
        bool(out["questions"].astype(str).str.endswith("?").all()) if len(out) else True,
    )
    return out


def save_prompt_csv_stratified(categorical_questions_df, harmfulqa_questions_df,
                                do_not_answer_df=None, beavertails_df=None,
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
                do_not_answer_df=do_not_answer_df,
                beavertails_df=beavertails_df,
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
    
    (
        categorical_questions,
        harmfulqa_questions,
        do_not_answer_questions,
        beavertails_questions,
    ) = load_harmful_datasets_separate()
    if (
        categorical_questions.empty
        and harmfulqa_questions.empty
        and do_not_answer_questions.empty
        and beavertails_questions.empty
    ):
        logger.error("No questions loaded from any dataset.")
        return False, pd.DataFrame(columns=['questions'])
    selected = select_prompts(
        categorical_questions, harmfulqa_questions,
        do_not_answer_df=do_not_answer_questions,
        beavertails_df=beavertails_questions,
        n_total=n_total, strategy=strategy, random_state=random_state
    )
    if selected.empty:
        logger.error("Selection produced 0 questions.")
        return False, selected
    success = save_prompt_csv_stratified(
        categorical_questions, harmfulqa_questions,
        do_not_answer_df=do_not_answer_questions,
        beavertails_df=beavertails_questions,
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
    parser = argparse.ArgumentParser(
        description=(
            "Build prompt.csv from CategoricalHarmfulQA, HarmfulQA, DoNotAnswer, "
            "and BeaverTails (prompts/questions ending with '?')."
        )
    )
    parser.add_argument("--n-total", type=int, default=500, help="Number of prompts to select (default: 500)")
    parser.add_argument("--strategy", choices=["stratified", "proportional", "union_sample"], default="stratified",
                        help="Selection strategy: stratified=equal share per dataset, proportional=by size, union_sample=dedupe then sample (default: stratified)")
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
        print(f"Ending with '?': {int(selected['questions'].astype(str).str.endswith('?').sum())}/{len(selected)}")
        print("=" * 50)
        print("Sample questions:")
        print(selected.sample(min(5, len(selected)), random_state=args.seed)[['questions']])
        print("=" * 50)
    else:
        logger.error("Failed to build prompt CSV.")
