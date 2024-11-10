from PIL import Image
import os
import pandas as pd

def lab_to_rgb_decimal(L, a, b):
 # Convert CIELAB to XYZ color space
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    # Apply inverse transformation
    x = 0.95047 * ((x ** 3) if (x ** 3) > 0.008856 else (x - 16 / 116) / 7.787)
    y = 1.00000 * ((y ** 3) if (y ** 3) > 0.008856 else (y - 16 / 116) / 7.787)
    z = 1.08883 * ((z ** 3) if (z ** 3) > 0.008856 else (z - 16 / 116) / 7.787)

    # Convert XYZ to RGB
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    # Apply gamma correction
    r = 1.055 * (r ** (1 / 2.4)) - 0.055 if r > 0.0031308 else 12.92 * r
    g = 1.055 * (g ** (1 / 2.4)) - 0.055 if g > 0.0031308 else 12.92 * g
    b = 1.055 * (b ** (1 / 2.4)) - 0.055 if b > 0.0031308 else 12.92 * b

    # Clip the values to be between 0 and 1
    r = max(0, min(1, r))
    g = max(0, min(1, g))
    b = max(0, min(1, b))

    return r, g, b

def rgb_to_hex(r, g, b):
    # Convert RGB decimal values to integers
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    
    # Convert to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def gen_rgb_images_from_cielab(lab_csv, output_dir):
    #using the cielab file generate the images
    file_name = 1
    with open(lab_csv) as f:
        for line in f:
            l, a, b = line.strip().split(',')
            l, a, b = float(l), float(a), float(b)
            rgb = lab_to_rgb_decimal(l, a, b)
            print("for file: ", file_name)
            print("rgb: ", rgb)
            print("lab: ", l, a, b)
            r, g, b = rgb
            hex = rgb_to_hex(r, g, b)
            print("rgb: ", rgb)
            print("hex: ", hex)
            img = Image.new("RGB", (64, 64), (int(r * 255), int(g * 255), int(b * 255)))
            img.save(os.path.join(output_dir, f"{file_name}.png"))
            file_name += 1

'''
# Example usage
L, a, b = 50.0, 28.891, -73.589
rgb = lab_to_rgb_decimal_v2(L, a, b)
print(rgb)  # Expected output: (0.18536, 0.43139, 0.96497)
'''
def get_words_and_associations(file):
    #read csv and first column is the word and the rest are the associations
    words = []
    associations_df = pd.read_csv(file)
    #print(associations_df)
    words = associations_df.iloc[:, 0].values
    associations = associations_df.iloc[:, 1:].values
    return words, associations

def compare_top_5_similarities(gt_similarities, predicted_similarities):
    # Compare the top 5 similarities
    #waiting for df's to be built to handle formatting of the output
    raise NotImplementedError