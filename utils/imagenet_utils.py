import pandas as pd
from pathlib import Path


def load_imagenet_classnames(csv_path: str = "data/imagenet_categories.csv"):
    """
    Загружает названия классов из CSV в порядке, совместимом с ImageFolder.

    Returns:
        classnames: List[str] — первые названия из колонки 'words' (до запятой)
        synset_to_category: Dict[str, str] — маппинг синсет → категория
        synset_to_words: Dict[str, str] — маппинг синсет → полное поле 'words'
    """
    df = pd.read_csv(csv_path)


    df = df.sort_values("synset").reset_index(drop=True)


    classnames = [
        words.split(",")[0].strip()
        for words in df["words"]
    ]

    synset_to_category = dict(zip(df["synset"], df["categories"]))
    synset_to_words = dict(zip(df["synset"], df["words"]))

    return classnames, synset_to_category, synset_to_words


def build_zero_shot_prompts(classnames, template="a sketch of a {}"):
    """
    Формирует текстовые шаблоны для нулевого шота.
    Для некоторых классов добавляем уточнения из категорий.
    """
    prompts = []
    for name in classnames:

        if name in ["crane", "dugong", "impala"]:
            prompts.append(f"a sketch of a {name} animal")
        elif name in ["organ", "schooner"]:
            prompts.append(f"a sketch of a {name} musical instrument")
        else:
            prompts.append(template.format(name))
    return prompts