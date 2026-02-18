"""
Data loading utilities for HuggingFace datasets and CSV sources.
Used for extracting and preparing prompt/question data for evolution.
"""

from datasets import load_dataset
import pandas as pd
import logging
import os
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_english_questions(ds, source_name, split_name):
    """
    Extract 'question' column from a HuggingFace datasets.Dataset object.
    - ds: HuggingFace Dataset object (already restricted to an English split)
    - source_name: Name of the source dataset.
    - split_name: Name of the split ('train', 'en', etc.)
    Returns: DataFrame with ['questions']
    """
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
        # Aggressively remove all types of quotes from questions during extraction
        def strip_all_quotes(text):
            """Remove all surrounding quotes (double, single, backticks, double single) from text"""
            if not isinstance(text, str):
                return text
            text = text.strip()
            # Remove quotes in multiple passes to handle nested quotes
            while True:
                original = text
                # Remove double quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text.strip('"').strip()
                # Remove single quotes
                if text.startswith("'") and text.endswith("'"):
                    text = text.strip("'").strip()
                # Remove backticks
                if text.startswith('`') and text.endswith('`'):
                    text = text.strip('`').strip()
                # Remove double single quotes
                if text.startswith("''") and text.endswith("''"):
                    text = text.strip("''").strip()
                # Stop if no more quotes to remove
                if text == original:
                    break
            return text
        df['questions'] = df['questions'].apply(strip_all_quotes)
        logger.info(f"Extracted {len(df)} questions from {source_name} ({split_name})")
        return df

    except Exception as e:
        logger.error(f"Error extracting questions from {source_name}: {e}")
        return pd.DataFrame(columns=['questions'])

def extract_harmfulqa_questions(ds, source_name, split_name):
    """
    Extract English questions from the HarmfulQA dataset format:
    The dataset has a "contexts" column which is a list of lists of questions,
    and "contexts_language" which includes a "en" split.
    """
    try:
        if "contexts" in ds.column_names:
            context_lists = ds["contexts"]
            all_questions = []
            for context_entry in context_lists:
                if isinstance(context_entry, (list, tuple)):
                    all_questions.extend([q for q in context_entry if isinstance(q, str)])
            df = pd.DataFrame({"questions": pd.Series(all_questions, dtype="string")})
            # Aggressively remove all types of quotes from questions during extraction
            def strip_all_quotes(text):
                """Remove all surrounding quotes (double, single, backticks, double single) from text"""
                if not isinstance(text, str):
                    return text
                text = text.strip()
                # Remove quotes in multiple passes to handle nested quotes
                while True:
                    original = text
                    # Remove double quotes
                    if text.startswith('"') and text.endswith('"'):
                        text = text.strip('"').strip()
                    # Remove single quotes
                    if text.startswith("'") and text.endswith("'"):
                        text = text.strip("'").strip()
                    # Remove backticks
                    if text.startswith('`') and text.endswith('`'):
                        text = text.strip('`').strip()
                    # Remove double single quotes
                    if text.startswith("''") and text.endswith("''"):
                        text = text.strip("''").strip()
                    # Stop if no more quotes to remove
                    if text == original:
                        break
                return text
            df['questions'] = df['questions'].apply(strip_all_quotes)
            logger.info(f"Extracted {len(df)} questions from 'contexts' in {source_name} ({split_name})")
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
    """
    Load and combine harmful question datasets from HuggingFace.
    Uses CategoricalHarmfulQA and HarmfulQA.
    Returns: DataFrame with unique questions from all datasets.
    """
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
    # Aggressively remove all types of quotes (double, single, backticks, double single)
    def strip_all_quotes(text):
        """Remove all surrounding quotes (double, single, backticks, double single) from text"""
        if not isinstance(text, str):
            return text
        text = text.strip()
        # Remove quotes in multiple passes to handle nested quotes
        while True:
            original = text
            # Remove double quotes
            if text.startswith('"') and text.endswith('"'):
                text = text.strip('"').strip()
            # Remove single quotes
            if text.startswith("'") and text.endswith("'"):
                text = text.strip("'").strip()
            # Remove backticks
            if text.startswith('`') and text.endswith('`'):
                text = text.strip('`').strip()
            # Remove double single quotes
            if text.startswith("''") and text.endswith("''"):
                text = text.strip("''").strip()
            # Stop if no more quotes to remove
            if text == original:
                break
        return text
    combined_df['questions'] = combined_df['questions'].apply(strip_all_quotes)
    combined_df = combined_df.dropna(subset=['questions'])
    combined_df = combined_df.drop_duplicates(subset=['questions'])

    logger.info(f"Combined unique questions: {len(combined_df)}")
    return combined_df

def load_harmful_datasets_separate():
    """
    Load harmful question datasets separately (for stratified sampling).
    Returns: Tuple of (categorical_questions_df, harmfulqa_questions_df)
    """
    categorical_questions = pd.DataFrame(columns=['questions'])
    harmfulqa_questions = pd.DataFrame(columns=['questions'])

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
        if categorical_questions.empty:
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
        if harmfulqa_questions.empty:
            logger.warning("HarmfulQA produced 0 questions after extraction.")
    except Exception as e:
        logger.error(f"Failed to load HarmfulQA dataset: {e}")

    # Helper function to aggressively remove all types of quotes
    def strip_all_quotes(text):
        """Remove all surrounding quotes (double, single, backticks, double single) from text"""
        if not isinstance(text, str):
            return text
        text = text.strip()
        # Remove quotes in multiple passes to handle nested quotes
        while True:
            original = text
            # Remove double quotes
            if text.startswith('"') and text.endswith('"'):
                text = text.strip('"').strip()
            # Remove single quotes
            if text.startswith("'") and text.endswith("'"):
                text = text.strip("'").strip()
            # Remove backticks
            if text.startswith('`') and text.endswith('`'):
                text = text.strip('`').strip()
            # Remove double single quotes
            if text.startswith("''") and text.endswith("''"):
                text = text.strip("''").strip()
            # Stop if no more quotes to remove
            if text == original:
                break
        return text
    
    # Clean each dataset separately
    if not categorical_questions.empty:
        categorical_questions['questions'] = categorical_questions['questions'].astype(str).str.strip()
        categorical_questions['questions'] = categorical_questions['questions'].apply(strip_all_quotes)
        categorical_questions = categorical_questions.dropna(subset=['questions'])
        categorical_questions = categorical_questions.drop_duplicates(subset=['questions'])
        logger.info(f"CategoricalHarmfulQA: {len(categorical_questions)} unique questions after cleaning")

    if not harmfulqa_questions.empty:
        harmfulqa_questions['questions'] = harmfulqa_questions['questions'].astype(str).str.strip()
        harmfulqa_questions['questions'] = harmfulqa_questions['questions'].apply(strip_all_quotes)
        harmfulqa_questions = harmfulqa_questions.dropna(subset=['questions'])
        harmfulqa_questions = harmfulqa_questions.drop_duplicates(subset=['questions'])
        logger.info(f"HarmfulQA: {len(harmfulqa_questions)} unique questions after cleaning")

    return categorical_questions, harmfulqa_questions

def save_questions_to_file(questions_df, filename=os.path.join("data", "harmful_questions.csv")):
    """
    Save questions DataFrame to CSV files.

    Args:
        questions_df: DataFrame with questions (should already be UNIQUE for prompt_extended.csv!)
        filename: Output filename (default: "data/harmful_questions.csv")
    """
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    success = True

    try:
        # Aggressively remove all types of quotes before saving
        def strip_all_quotes(text):
            """Remove all surrounding quotes (double, single, backticks, double single) from text"""
            if not isinstance(text, str):
                return text
            text = text.strip()
            # Remove quotes in multiple passes to handle nested quotes
            while True:
                original = text
                # Remove double quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text.strip('"').strip()
                # Remove single quotes
                if text.startswith("'") and text.endswith("'"):
                    text = text.strip("'").strip()
                # Remove backticks
                if text.startswith('`') and text.endswith('`'):
                    text = text.strip('`').strip()
                # Remove double single quotes
                if text.startswith("''") and text.endswith("''"):
                    text = text.strip("''").strip()
                # Stop if no more quotes to remove
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
        # Aggressively remove all quotes before saving
        def strip_all_quotes(text):
            """Remove all surrounding quotes from text"""
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

    # Note: prompt.csv is now handled separately by save_prompt_csv_stratified()
    # to ensure 250 questions from each dataset

    return success

def save_prompt_csv_stratified(categorical_questions_df, harmfulqa_questions_df, 
                                n_per_dataset=250, random_state=48, filename="data/prompt.csv"):
    """
    Create prompt.csv by sampling n_per_dataset questions from each dataset separately.
    
    Args:
        categorical_questions_df: DataFrame with questions from CategoricalHarmfulQA
        harmfulqa_questions_df: DataFrame with questions from HarmfulQA
        n_per_dataset: Number of questions to sample from each dataset (default: 250)
        random_state: Random seed for reproducibility (default: 48)
        filename: Output filename (default: "data/prompt.csv")
    
    Returns:
        bool: True if successful, False otherwise
    """
    outdir = os.path.dirname(filename)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    try:
        # Sample from each dataset separately
        categorical_sample = categorical_questions_df.sample(
            n=min(n_per_dataset, len(categorical_questions_df)), 
            random_state=random_state
        ) if not categorical_questions_df.empty else pd.DataFrame(columns=['questions'])
        
        harmfulqa_sample = harmfulqa_questions_df.sample(
            n=min(n_per_dataset, len(harmfulqa_questions_df)), 
            random_state=random_state
        ) if not harmfulqa_questions_df.empty else pd.DataFrame(columns=['questions'])
        
        # Combine samples
        combined_sample = pd.concat([categorical_sample, harmfulqa_sample], ignore_index=True)
        
        # Aggressively remove all quotes before saving
        def strip_all_quotes(text):
            """Remove all surrounding quotes from text"""
            if not isinstance(text, str):
                return text
            text = text.strip()
            while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text.strip('"').strip("'").strip()
            return text
        combined_sample['questions'] = combined_sample['questions'].apply(strip_all_quotes)
        
        # Save to file (QUOTE_MINIMAL only quotes when necessary, but we've already stripped quotes from text)
        combined_sample[['questions']].to_csv(filename, index=False, header=True, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Saved {len(combined_sample)} questions to {filename} "
                   f"({len(categorical_sample)} from CategoricalHarmfulQA, "
                   f"{len(harmfulqa_sample)} from HarmfulQA)")
        return True
    except Exception as e:
        logger.error(f"Failed to save stratified prompt.csv: {e}")
        return False

    return success

def get_questions_as_list(questions_df):
    """
    Extract questions as a simple list for easy use in other modules.

    Args:
        questions_df: DataFrame with questions

    Returns:
        List of question strings
    """
    return questions_df['questions'].tolist()

if __name__ == "__main__":
    # Load datasets separately for stratified sampling
    categorical_questions, harmfulqa_questions = load_harmful_datasets_separate()
    
    # Also load combined for harmful_questions.csv and prompt_extended.csv
    all_questions = load_harmful_datasets()

    if not all_questions.empty:
        logger.info(f"Number of unique questions (combined): {len(all_questions)}")
        
        # Save full datasets (harmful_questions.csv and prompt_extended.csv)
        saved = save_questions_to_file(all_questions)
        
        # Save stratified prompt.csv (250 from each dataset)
        prompt_saved = save_prompt_csv_stratified(
            categorical_questions, 
            harmfulqa_questions, 
            n_per_dataset=250, 
            random_state=48
        )
        
        if saved and prompt_saved:
            print("="*50)
            print("Saved questions to:")
            print("  - data/harmful_questions.csv (all unique questions)")
            print("  - data/prompt_extended.csv (all unique questions)")
            print("  - data/prompt.csv (250 from each dataset = 500 total)")
            print("="*50)
            print("SAMPLE QUESTIONS:")
            print("="*50)
            print(all_questions.sample(min(5, len(all_questions))))
            print("="*50)
            print(f"TOTAL UNIQUE QUESTIONS: {len(all_questions)}")
            print(f"CategoricalHarmfulQA: {len(categorical_questions)} unique questions")
            print(f"HarmfulQA: {len(harmfulqa_questions)} unique questions")
            print("="*50)
        else:
            logger.error("Failed to save some files.")
    else:
        logger.error("No questions were loaded!")
