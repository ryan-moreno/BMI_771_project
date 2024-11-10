from PIL import Image
import colorsys
import numpy as np
import pandas as pd
import os

def lab_to_rgb_decimal(L, a, b):
    # Convert CIELAB to XYZ color space
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    # Apply inverse transformation
    x = 0.95047 * ((x ** 3) if (x ** 3 > 0.008856) else ((x - 16/116) / 7.787))
    y = 1.00000 * ((y ** 3) if (y ** 3 > 0.008856) else ((y - 16/116) / 7.787))
    z = 1.08883 * ((z ** 3) if (z ** 3 > 0.008856) else ((z - 16/116) / 7.787))

    # Convert XYZ to RGB
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    return (r, g, b)


#gen a function for getting the CIELAB values of a color from a pixel's RGB values
def rgb_to_lab(r, g, b):
    # Convert RGB to a range of [0, 1]
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Apply sRGB gamma correction
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # Convert to XYZ space
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # Normalize for D65 white point
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    # Apply XYZ to LAB conversion
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16 / 116)

    l = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b = 200 * (f(y) - f(z))

    return (l, a, b)

def rgb_to_hex(r, g, b):
    hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return hex

def hex_to_rgb(hex_code):
    """Convert hex color to RGB tuple."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def generate_image_from_hex(hex_code, filename="output_image.png"):
    # Convert hex to RGB
    rgb_color = hex_to_rgb(hex_code)
    
    # Generate image
    img = Image.new("RGB", (64, 64), rgb_color)
    img.save(filename)
    
    # Convert RGB to CIELAB
    lab_color = rgb_to_lab(*rgb_color)
    
    return {
        "rgb": rgb_color,
        "lab": lab_color,
        "filename": filename
    }
    
def gen_img_from_lab(l, a, b, size=(64, 64)):
    # Generate image
    rgb_color = lab_to_rgb_decimal(l, a, b)
    
    r, g, b = rgb_color
    print(rgb_color)
    
    # Clip and scale to 8-bit color
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)
    
    rgb_color = (r, g, b)
    print(rgb_color)
    img = Image.new("RGB", size, rgb_color)
    return img
    
def gen_images_from_csv(output_dir, csv_cielab):
    #using the cielab file generate the images
    with open(csv_cielab) as f:
        # Read the file every row has 3 values L, A, B separated by commas
        line_index = 0
        for line in f:
            name = line_index+1
            l, a, b = line.split(',')
            # make sure l, a, b are floats
            l, a, b = float(l), float(a), float(b)
            img = gen_img_from_lab(l, a, b)
            img.save(os.path.join(output_dir, f"{name}.png"))
            line_index += 1
    print("Images generated")
        
def check_img_colors(input_dir, csv_path, csv_rgb):
    #for every image in the input directory
    #check if the colors match the ones in the csv file
    #if they do not match, print out the image name
    #and the color that does not match
    #if they do match, print out the image name
    #and "colors match"
    rgbs_to_compare = pd.read_csv(csv_rgb)
    print(rgbs_to_compare.head())
    hexes_to_compare = pd.read_csv(csv_path)
    print(hexes_to_compare.head())
    #use os to loop through the files in the input directory
    for file in os.listdir(input_dir):
        img = Image.open(os.path.join(input_dir, file))
        rgb = img.getpixel((0, 0))
        hex_code = rgb_to_hex(*rgb)
        #get the name of the file
        name = file.split('.')[0]
        # the name could be converted to an int and then used for lookup in hexes_to_compare
        # if the name is an int
        if name.isdigit():
            name = int(name)
        #get the hex code from the csv file
        hex_code_compare = hexes_to_compare[hexes_to_compare['index'] == name]['hex']
        # the rgb value does not have a convenient index the order of the rows in the csv file is the index with should align with name-1 since the name is 1 indexed
        rgb_compare = rgbs_to_compare.iloc[name-1]
        r_compare, g_compare, b_compare = int(max(0, min(1, rgb_compare[0])) * 255), int(max(0, min(1, rgb_compare[1])) * 255), int(max(0, min(1, rgb_compare[2])) * 255)
        #check if the colors match
        print(f"Checking image {name}")
        print(f"Expected: {hex_code_compare}, {rgb_compare}")
        print(f"Got: {hex_code}, {rgb}")
        if hex_code != hex_code_compare or rgb != (r_compare, g_compare, b_compare):
            print(f"Image {name} does not match the colors in the csv file")
            print(f"Expected: {hex_code_compare}, {rgb_compare}")
            print(f"Got: {hex_code}, {rgb}")
        else:
            print(f"Image {name} colors match")