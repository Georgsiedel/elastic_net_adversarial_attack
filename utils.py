from robustbench.utils import load_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
import foolbox as fb
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
import torch
import torchvision
import torchvision.transforms as transforms
def load_dataset(dataset, dataset_split):

    if dataset== 'imagenet':

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224)
                                    ])
        
        testset = datasets.ImageFolder(root=f'../datasets/ImageNet/val', transform=transform)
        
    elif dataset == 'cifar10':

        transform = transforms.Compose([
        transforms.ToTensor(),
                                    ])
        
        testset = datasets.CIFAR10(root='./data/cifar', train=False, download=True, transform=transform)

    else: 
        raise KeyError("Dataset not implemented.")
    
    # Truncated testset for experiments and ablations
    if isinstance(dataset_split, int):
        testset, _ = torch.utils.data.random_split(testset,
                                                          [dataset_split, len(testset) - dataset_split],
                                                          generator=torch.Generator().manual_seed(42))
    
    # Extract data and labels from torchvision dataset
    xtest = torch.stack([data[0] for data in testset])
    ytest = torch.tensor([data[1] for data in testset])

    return xtest, ytest

def get_model(dataset, modelname, norm=None):
    if modelname=='CroceL1' and dataset=='cifar10': 
        '''
        based on https://github.com/fra31/robust-finetuning
        "Adversarial robustness against multiple and single lp-threat models via quick fine-tuning of robust classifiers", Francesco Croce, Matthias Hein, ICML 2022
        https://arxiv.org/abs/2105.12508
        '''
        from models import fast_models
        net = fast_models.PreActResNet18(10, activation='softplus1', cuda=cuda)
        ckpt = torch.load('./models/pretrained_models/CroceL1.pth', map_location=device)
        net.load_state_dict(ckpt)
    elif modelname in ['MainiMSD', 'MainiAVG'] and dataset == 'cifar10':
        '''
        based on https://github.com/locuslab/robust_union/tree/master/CIFAR10
        "Adversarial Robustness Against the Union of Multiple Perturbation Models", by Pratyush Maini, Eric Wong and Zico Kolter, ICML 2020
        https://arxiv.org/abs/2105.12508
        '''
        from models import preact_resnet
        net = preact_resnet.PreActResNet18()
        ckpt = torch.load(f'./models/pretrained_models/{modelname}.pt', map_location=device)
        net.load_state_dict(ckpt)
    else: #robustbench models
        net = load_model(model_name=modelname, dataset=dataset, threat_model=norm) #'Wang2023Better_WRN-28-10'
        modelname = modelname + '_' + norm
        
    #net = torch.nn.DataParallel(net)
    net.eval()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    if dataset == 'imagenet':
        nb_classes = 1000 
        input_shape=(3, 224, 224)
    elif dataset == 'cifar10':
        nb_classes = 10
        input_shape=(3, 32, 32)
    else: 
        raise KeyError("Dataset not implemented.")

    # Initialize wrappers for ART toolbox and foolbox
    art_net = PyTorchClassifier(model=net,
                               loss=criterion,
                               optimizer=optimizer,
                               input_shape=input_shape,
                               nb_classes=nb_classes,
                               device_type=device,
                               clip_values=(0.0, 1.0))
    fb_net = fb.PyTorchModel(net, bounds=(0.0, 1.0), device=device)

    net.to(device)

    return net, art_net, fb_net, modelname


def test_accuracy(model, xtest, ytest):
    model.eval()
    correct, total = 0, 0
    batch_size = 100

    with torch.no_grad():
        for i in range(0, len(xtest), batch_size):
            x_batch = xtest[i:i + batch_size].to(device)
            y_batch = ytest[i:i + batch_size].to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    model.to(device)

    accuracy = (correct / total) * 100
    print(f'\nAccuracy of the test set is: {accuracy:.3f}%\n')
