import argparse
import os

from PIL import Image


def validate(path):
    image = Image.open(path)

    pixels = image.load()

    non_white = []

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if pixels[x, y][0] != 255 or pixels[x, y][1] != 255 or pixels[x, y][2] != 255:
                non_white.append((x, y))

    print(f'There are {len(non_white)} non-white pixels on this image. {path}')

    if len(non_white) > 0:
        print('This image is valid.')

    return True


def validate_tensor(tensor):
    with os.open('C:/Users/KIUser/Desktop/elastic_net_adversarial_attack/result', flags=os.O_RDWR) as f:
        f.write(str(tensor))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/attack_comparison_standard_cifar10_images/exp_attack_l1_2_delta.png',help='Path of image to validate')
    args = parser.parse_args()

    validate(args.path)
