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

    fun_handles = {
        "gen_images": gen_images,
    }
    run_tests(args.function_name, fun_handles)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_model_name(model="ViT-B/32"):
    # List of different CLIP models you can use
    clip_models = {
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-L/14": "openai/clip-vit-large-patch14",
        "RN50": "openai/clip-rn50",
        "RN101": "openai/clip-rn101",
        # Add more models as needed
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
    key_words, gt_assoc = get_words_and_associations()
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
        print(tentative_associations)
        for i in range(len(key_words)):
            for j in range(len(tentative_associations)):
                percentage = round(tentative_associations[j], ndigits=2)
                text_prompts.append(
                    f"The color of the image is associated with the concept {key_words[i]} with an association strength of {percentage} out of 1."
                )
        folder_to_save = "./output/text_prompts/test_4"
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

        print(f"Image: {image}")
        print("Cosine Similarities:", cosine_similarities)

    # Label indices of results_df with key words
    key_words, _ = get_words_and_associations()
    assert len(key_words) == len(results_df)

    results_df.index = key_words

    folder_to_save = f"./output/cosine_similarities/test_{test_num}_scores/"
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    file_name = os.path.join(folder_to_save, "similarity_scores_df.txt")
    results_df.to_csv(file_name, index=True)
    print("Cosine Similarities saved to file.")


def get_predictions(prompt_style=1):
    """
    Gets the models predicted associations

    Inputs:
        prompt_style: The prompt style to get predictions for

    Outputs:
        output: data frame (rows are words, columns are color patches)
    """
    cosine_similarity_folder = f"output/cosine_similarities/test_{prompt_style}_scores"
    file_name = os.path.join(cosine_similarity_folder, "similarity_scores_df.txt")
    pred = pd.read_csv(file_name, index_col=0)
    return pred


def evaluate_model(prompt_style=1):
    """
    Evaluates the model's labels against the ground truth associations

    Inputs:
        prompt_style: The prompt style to evaluate
    """
    key_words, _ = get_words_and_associations(as_data_frame=False)
    gt = get_words_and_associations(as_data_frame=True)
    pred = get_predictions(prompt_style)

    for key_word in key_words:
        gt_rankings = gt.loc[key_word].sort_values(ascending=False)
        pred_rankings = pred.loc[key_word].sort_values(ascending=False)
        ranking_df = pd.DataFrame(columns=["color_index", "gt_ranking", "pred_ranking"])
        ranking_df["color_index"] = range(1, NUM_COLOR_PATCHES + 1)
        ranking_df["gt_ranking"] = gt_rankings.index
        ranking_df["pred_ranking"] = pred_rankings.index
        top_10_overlap = len(
            set(gt_rankings.index[:10]).intersection(pred_rankings.index[:10])
        )
        print(f"Top 10 overlap for {key_word}: {top_10_overlap}")
    print(gt)
    print(pred)
    pass


def plot_color_rankings(sorted_colors_indices):
    """
    Plots the color rankings

    Inputs:
        sorted_colors_indices: The sorted indices of the colors, with the 1st being the most associated color
    """

    pass


if __name__ == "__main__":
    evaluate_model(prompt_style=1)
    # perform_test(test_num=1)
    # generate_text_prompts(test=4)
    # gen_imgs()
    # get_words_and_associations()
    # check_file_colors()
    # main_func()
