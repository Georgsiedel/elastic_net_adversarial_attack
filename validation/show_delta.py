import argparse
import os
import matplotlib.pyplot as plt
import numpy
from PIL import Image

def show_delta_text(path):
    image = Image.open(path)

    pixels = image.load()

    os.remove('./result')

    with open('./result', 'w') as f:
        for x in range(image.size[0]):
            output = ''

            for y in range(image.size[1]):
                output += f'\t{abs(255 - pixels[x, y][0]) + abs(255 - pixels[x, y][1] + abs(255 - pixels[x, y][2]))}'

            f.write(f'\t{output}\n')


def show_delta_heatmap(path):
    image = Image.open(path)

    #delta_list = image_to_delta_list(image)

    plt.imshow(image, cmap='hot', interpolation='nearest')

    plt.show()


def image_to_delta_list(image):
    result = []

    pixels = image.load()

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            result.append(abs(255 - pixels[x, y][0]) + abs(255 - pixels[x, y][1]) + abs(255 - pixels[x, y][2]))

    return result
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/attack_comparison_standard_cifar10_images/exp_attack_l1_2_delta.png',help='Path of image to validate')
    parser.add_argument('--type', type=str, default='heatmap', choices={"text", "heatmap"})
    args = parser.parse_args()

    if args.type == "text":
        show_delta_text(args.path)
    else:
        show_delta_heatmap(args.path)
