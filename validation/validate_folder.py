from validate_image import validate
import argparse
import glob


def validate_folder(path):
    files = glob.glob(path + '/*delta*')

    for i in range(len(files)):
        valid = validate(files[i])

        if not valid:
            print(f'File \'{files[i]}\' is not valid.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default='data/attack_comparison_standard_cifar10_images',
                        help='Path of image to validate')
    args = parser.parse_args()

    validate_folder(args.path)

