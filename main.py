import numpy as np
import pandas as pd
import os
import argparse
from runTests import run_tests
import helper_func_2 as hf2
from PIL import Image
import torch.nn.functional as F
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr

NUM_COLOR_PATCHES = 71


def main_func():
    parser = argparse.ArgumentParser(
        description="Execute a specific test or all tests."
    )
    parser.add_argument(
        "function_name",
        type=str,
        nargs="?",
        default="all",
        help='Name of the function to test or "all" to execute all the registered functions',
    )
    args = parser.parse_args()

    fun_handles = {}
    run_tests(args.function_name, fun_handles)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_model_name(model="ViT-B/32"):
    # List of different CLIP models you can use
    clip_models = {
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-L/14": "openai/clip-vit-large-patch14",
        # Add more models as needed
        # maybe try this from microsoft https://huggingface.co/microsoft/Phi-3.5-vision-instruct
        # and this from microsoft https://huggingface.co/microsoft/Florence-2-large
    }
    # Load the CLIP model
    model_name = clip_models[model]
    return model_name


def encode_image(image_path, processor, model):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")  # Preprocess
    inputs = inputs.to(device)
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)  # Get image embeddings
    return image_embeddings


def encode_text_and_compute_similarity(text_prompts, image_encoding, processor, model):
    # Encode text prompts
    inputs = processor(
        text=text_prompts, return_tensors="pt", padding=True, truncation=True
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)

    # normalize the image encoding and text embeddings (L2 normalization) TODO: this may not be needed because I believe CLIP does this already
    image_encoding = F.normalize(image_encoding, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute cosine similarity between text and image embeddings
    cosine_similarities = F.cosine_similarity(
        text_embeddings, image_encoding.repeat(text_embeddings.shape[0:1] + (1,))
    )
    return cosine_similarities


def read_text_prompts(test=1):
    if test == 1:
        file_name = os.path.join("./output/text_prompts/test_1", "text_prompts.txt")
    elif test == 2:
        file_name = os.path.join("./output/text_prompts/test_2", "text_prompts.txt")
    elif test == 3:
        file_name = os.path.join("./output/text_prompts/test_3", "text_prompts.txt")
    elif test == 4:
        file_name = os.path.join("./output/text_prompts/test_4", "text_prompts.txt")
    elif test == 5:
        file_name = os.path.join("./output/text_prompts/test_5", "text_prompts.txt")
    else:
        raise ValueError("Invalid test number.")
    text_prompts = []
    with open(file_name) as f:
        for line in f:
            text_prompts.append(line.strip())
    # print(text_prompts)
    return text_prompts


"""
image_path = 'your_image.jpg'
image_encoding = encode_image(image_path)
print("Image Encoding Shape:", image_encoding.shape)
"""


def gen_imgs():
    cielab_csv = os.path.join("data", "UW71lab.csv")
    output_dir = "output/images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hf2.gen_rgb_images_from_cielab(cielab_csv, output_dir)


def get_words_and_associations(as_data_frame=False):
    """
    Get ground truth annotations

    Inputs:
        as_data_frame: If True, return as a pandas DataFrame; otherwise, return as a tuple of words and associations

    Outputs:
        output: Dataframe or tuple of words and associations (see as_data_frame parameter)
    """
    source_file = os.path.join("data", "uw_71_ratings_matrix.csv")
    words, associations = hf2.get_words_and_associations(source_file)

    if as_data_frame:
        return pd.DataFrame(
            associations, index=words, columns=range(1, NUM_COLOR_PATCHES + 1)
        )
    else:
        return words, associations


def generate_text_prompts(test=1):
    key_words, _ = get_words_and_associations()
    text_prompts = []
    if test == 1:
        for i in range(len(key_words)):
            text_prompts.append(
                f"The color of the image represents the word {key_words[i]}."
            )
        folder_to_save = "./output/text_prompts/test_1"
    elif test == 2:
        for i in range(len(key_words)):
            text_prompts.append(
                f"The concept {key_words[i]} is well represented by the color of the image."
            )
        folder_to_save = "./output/text_prompts/test_2"
    elif test == 3:
        for i in range(len(key_words)):
            text_prompts.append(
                f"The color of the image is associated with the concept {key_words[i]}."
            )
        folder_to_save = "./output/text_prompts/test_3"
    elif test == 4:
        # this one will be different in that every concept will have multiple prompts
        # for 0 to 1 in .1 increments
        tentative_associations = np.linspace(0, 1, 11)
        # print(tentative_associations)
        for i in range(len(key_words)):
            for j in range(len(tentative_associations)):
                percentage = round(tentative_associations[j], ndigits=2)
                text_prompts.append(
                    f"The color of the image is associated with the concept {key_words[i]} with an association strength of {percentage} out of 1."
                )
        folder_to_save = "./output/text_prompts/test_4"
    elif test == 5:
        # this test is for debugging purposes I will use a color in the sentence to see if it leads to a higher similarity score
        text_prompts.append(f"The color of the image is yellow.")
        folder_to_save = "./output/text_prompts/test_5"
    else:
        raise ValueError("Invalid test number.")
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    # Save the text prompts to a file, one line per prompt
    # Ensure the directory exists
    os.makedirs(folder_to_save, exist_ok=True)

    # Save the text prompts to a file
    # generate a file to update
    file_name = os.path.join(folder_to_save, "text_prompts.txt")
    # create the file
    with open(file_name, "w") as f:
        for prompt in text_prompts:
            f.write(prompt + "\n")

    # return text_prompts


def perform_test(test_num=1, which_model="ViT-B/32", images_path="./output/images/"):
    # Check and set device to MPS
    print(f"Using device: {device}")
    # generate a folder name based on the model name but without the slashes or dashes
    model_folder = which_model.replace("/", "_").replace(
        "-", "_"
    )  # this will be used to save the data to the correct folder later

    model_name = get_model_name(which_model)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    text_prompts = read_text_prompts(test_num)

    results_df = pd.DataFrame(columns=list(map(str, range(1, NUM_COLOR_PATCHES + 1))))

    # get each image in the images folder
    for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        image_name = image.split(".")[0]
        image_encoding = encode_image(image_path, processor=processor, model=model)

        cosine_similarities = encode_text_and_compute_similarity(
            text_prompts, image_encoding, processor=processor, model=model
        )

        results_df[image_name] = cosine_similarities.cpu().numpy().flatten()

        # print(f"Image: {image}")
        # print("Cosine Similarities:", cosine_similarities)

    # Label indices of results_df with key words
    key_words, _ = get_words_and_associations()
    assert len(key_words) == len(results_df)

    results_df.index = key_words

    folder_to_save = (
        f"./output/cosine_similarities/{model_folder}/test_{test_num}_scores/"
    )
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    file_name = os.path.join(folder_to_save, "similarity_scores_df.txt")
    results_df.to_csv(file_name, index=True)
    print("Cosine Similarities saved to file.")


def get_predictions(prompt_style=1, which_model="ViT-B/32"):
    """
    Gets the models predicted associations

    Inputs:
        prompt_style: The prompt style to get predictions for

    Outputs:
        output: data frame (rows are words, columns are color patches)
    """
    model_folder = which_model.replace("/", "_").replace("-", "_")
    cosine_similarity_folder = (
        f"output/cosine_similarities/{model_folder}/test_{prompt_style}_scores"
    )
    file_name = os.path.join(cosine_similarity_folder, "similarity_scores_df.txt")
    pred = pd.read_csv(file_name, index_col=0)
    pred.columns = pred.columns.astype(int)
    return pred


def plot_scores(score_series, plot_path, word, name):
    """
    Plots the rankings of the predicted associations according to scores
    """

    score_df = score_series.to_frame(name="score")
    colors = pd.read_csv("data/UW_71_color_dict.csv", index_col=0)
    score_df = pd.DataFrame(score_df)
    score_df.columns = ["score"]
    score_df["color_index"] = score_df.index
    score_df = score_df.merge(colors, left_on="color_index", right_on="index")
    score_df["color_index"] = score_df["color_index"].astype(str)
    score_df = score_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Bar plot with colors
    plt.figure(figsize=(8, 8))
    plt.bar(
        score_df["color_index"],
        score_df["score"],
        color=score_df["hex"],
        label="Score",
    )
    plt.xlabel("Color")
    plt.ylabel(f"Score")
    plt.xticks([])
    plt.title(f"{name} scores for {word}")
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()


def compare_rankings_plot(ranking_df, plot_path_prefix, word):
    """
    Plots the comparison of the rankings of the ground truth and predicted rankings.
    Creates two plots: one ordered by GT and one ordered by Pred

    Inputs:
        ranking_df: The data frame containing the rankings (columns of "color_index", "gt_ranking", and "pred_ranking")
        plot_path: Path to save the rankings plot to
    """
    colors = pd.read_csv("data/UW_71_color_dict.csv", index_col=0)
    ranking_df = ranking_df.merge(colors, left_on="color_index", right_on="index")

    ## Plot, ordered by GT
    bar_heights = NUM_COLOR_PATCHES - ranking_df["pred_ranking"].astype(int)
    plt.figure(figsize=(8, 8))
    plt.bar(
        ranking_df["gt_ranking"].astype(int),
        bar_heights,
        bottom=ranking_df["pred_ranking"].astype(int),
        color=ranking_df["hex"],
        label="Rank Comparison",
    )
    plt.plot(
        [1, 71],
        [1, 71],
        color="black",
        linestyle="--",
        label="Perfect Match (Diagonal)",
    )
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("Ground Truth Rankings")
    plt.ylabel("Predicted Rankings")
    plt.title(f"Comparison of rankings for {word} (colored by ground truth rankings)")
    plt.legend()
    plt.savefig(f"{plot_path_prefix}_by_gt.png", format="png", dpi=300)
    plt.close()

    ## Plot, ordered by Pred
    bar_lengths = NUM_COLOR_PATCHES - ranking_df["gt_ranking"].astype(int)
    plt.figure(figsize=(8, 8))
    plt.barh(
        ranking_df["pred_ranking"].astype(int),
        bar_lengths,
        left=ranking_df["gt_ranking"].astype(int),
        color=ranking_df["hex"],
        label="Rank Comparison",
    )
    plt.plot(
        [1, 71],
        [1, 71],
        color="black",
        linestyle="--",
        label="Perfect Match (Diagonal)",
    )

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("Ground Truth Rankings")
    plt.ylabel("Predicted Rankings")
    plt.title(f"Comparison of rankings for {word} (colored by predicted rankings)")
    plt.legend()
    plt.savefig(f"{plot_path_prefix}_by_pred.png", format="png", dpi=300)
    plt.close()


def evaluate_model(prompt_style=1, which_model="ViT-B/32"):
    """
    Evaluates the model's labels against the ground truth associations

    Inputs:
        prompt_style: The prompt style to evaluate
    """
    model_name = which_model.replace("/", "_").replace("-", "_")
    eval_folder = f"./output/evaluation/{model_name}/test_{prompt_style}_eval/"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    key_words, _ = get_words_and_associations(as_data_frame=False)
    gt = get_words_and_associations(as_data_frame=True)
    pred = get_predictions(prompt_style, which_model)

    for key_word in key_words:
        plot_scores(
            gt.T[key_word],
            os.path.join(eval_folder, f"{key_word}_gt_scores.png"),
            key_word,
            "Ground truth",
        )
        plot_scores(
            pred.T[key_word],
            os.path.join(eval_folder, f"{key_word}_pred_scores.png"),
            key_word,
            "Predicted",
        )
        # Get the rankings of the ground truth and predicted associations
        gt_rankings = gt.loc[key_word].sort_values(ascending=False)
        pred_rankings = pred.loc[key_word].sort_values(ascending=False)
        ranking_df = pd.DataFrame(
            {
                "color_index": range(1, NUM_COLOR_PATCHES + 1),
                "gt_ranking": [
                    gt_rankings.index.get_loc(color_index) + 1
                    for color_index in range(1, NUM_COLOR_PATCHES + 1)
                ],
                "pred_ranking": [
                    pred_rankings.index.get_loc(color_index) + 1
                    for color_index in range(1, NUM_COLOR_PATCHES + 1)
                ],
            }
        )

        ranking_plot_path = os.path.join(eval_folder, f"{key_word}_ranking_plot")
        compare_rankings_plot(ranking_df, ranking_plot_path, key_word)


def evaluation_metrics(prompt_style=1, which_model="ViT-B/32"):
    """
    Evaluates the model's performance using various metrics
    code for the metrics will be added to the helper_func_2.py file
    We will evaluate the model using the following metrics:
    - Pearson Correlation Coefficient (PCC)
    - Total Variation (TV)
    - Earth Mover's Distance (EMD)
    - Entropy Distance (ED)
    - Spearman Rank Correlation (SRC)
    """
    model_folder = which_model.replace("/", "_").replace("-", "_")
    output_folder = (
        f"./output/evaluation_metrics/{model_folder}/test_{prompt_style}_metrics"
    )
    gt = get_words_and_associations(as_data_frame=True)
    pred = get_predictions(prompt_style, which_model)

    pearson_correlations = []
    total_variances = []
    earth_movers_distances = []
    entropy_distances = []
    spearman_rank_correlations = []

    for word in gt.index:
        gt_values = gt.loc[word].values
        pred_values = pred.loc[word].values
        # Calculate Pearson Correlation
        r, _ = pearsonr(gt_values, pred_values)
        pearson_correlations.append(r)

        # Calculate Total Variation
        # half of the L1-distance between two distributions
        tv = 0.5 * np.sum(np.abs(gt_values - pred_values))
        total_variances.append(tv)

        # Calculate Earth Mover's Distance
        emd = wasserstein_distance(gt_values, pred_values)
        earth_movers_distances.append(emd)

        # Entropy Distance ED
        # ED(pˆ,r) = abs( sum (pˆi log ˆpi) − sum (ri logri) )
        gt_log_values = np.array([v * np.log(v) if v > 0 else 0 for v in gt_values])
        pred_log_values = np.array([v * np.log(v) if v > 0 else 0 for v in pred_values])
        ed = np.abs(gt_log_values.sum() - pred_log_values.sum())
        entropy_distances.append(ed)

        # Spearman Rank Correlation
        src, _ = spearmanr(gt_values, pred_values)
        spearman_rank_correlations.append(src)

        # validate order by printing the values
        # print("for word: ", word, "gt_values: ", gt_values, "pred_values: ", pred_values)
        # print("PCC: ", r, "SRC: ", src, "TV: ", tv, "EMD: ", emd, "ED: ", ed)

    # create a data frame to store the results
    evaluation_metrics_df = pd.DataFrame()
    evaluation_metrics_df["word"] = gt.index
    evaluation_metrics_df["PCC"] = pearson_correlations
    evaluation_metrics_df["SRC"] = spearman_rank_correlations
    evaluation_metrics_df["TV"] = total_variances
    evaluation_metrics_df["EMD"] = earth_movers_distances
    evaluation_metrics_df["ED"] = entropy_distances

    # Save the evaluation metrics to a file
    # Ensure the directory exists
    os.makedirs(output_folder, exist_ok=True)

    file_name = os.path.join(output_folder, "evaluation_metrics_df.txt")
    evaluation_metrics_df.to_csv(file_name, index=True)
    print("Evaluation Metrics saved to file.")
    pass


if __name__ == "__main__":
    # evaluation_metrics(prompt_style=1, which_model="ViT-B/32")
    evaluate_model(prompt_style=1, which_model="ViT-B/32")
    # perform_test(test_num=3, which_model="ViT-B/32")
    # generate_text_prompts(test=4)
    # gen_imgs()
    # get_words_and_associations()
    # check_file_colors()
    # main_func()
