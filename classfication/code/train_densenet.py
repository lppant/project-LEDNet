import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from utils import train, evaluate
from plots import plot_learning_curves
from read import LednetDataSet
import torchvision.transforms as transforms
import argparse

NUM_EPOCHS = 9  # changed for plot
BATCH_SIZE = 50
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.
PATH_OUTPUT = ""

def get_arguments():
    '''
    Get the arguments for etl
    :return:
    '''
    parser = argparse.ArgumentParser(description="train densenet 121 with 9 epocs and batch size of 50")
    parser.add_argument("-p", required=True, dest="prefix_path", help="Path to directory containing image dataset")
    return parser.parse_args()

def preprocess():
    # Set a correct path to the seizure data file you downloaded
    args = get_arguments()
    PATH_TRAIN_FILE = args.prefix_path + "/train"
    PATH_VALID_FILE = args.prefix_path + "/val"
    PATH_TEST_FILE = args.prefix_path + "/test"

    # Path for saving model
    PATH_OUTPUT = args.prefix_path + "/output/densenet/"
    os.makedirs(PATH_OUTPUT, exist_ok=True)


    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_dataset = LednetDataSet(data_dir=PATH_TRAIN_FILE,
                                     image_list_file=PATH_TRAIN_FILE + '/train.csv',
                                    transform=transforms.Compose([
                                    transforms.Resize([256, 256]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
                                     )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,
                                               shuffle=False,num_workers=8,pin_memory=True)
    #train_loader = load_densenet_dataset(PATH_TRAIN_FILE,BATCH_SIZE,NUM_WORKERS,True)

    valid_dataset = LednetDataSet(data_dir=PATH_VALID_FILE,
                                     image_list_file=PATH_VALID_FILE + '/val.csv',
                                     transform=transforms.Compose([
                                         transforms.Resize([256, 256]),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ])
                                     )

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,
                                               shuffle=False,num_workers=8,pin_memory=True)


    test_dataset = LednetDataSet(data_dir=PATH_TEST_FILE,
                                     image_list_file=PATH_TEST_FILE + '/test.csv',
                                    transform=transforms.Compose([
                                        transforms.Resize([256, 256]),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                    ])
                                     )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,
                                               shuffle=False,num_workers=8,pin_memory=True)

    return train_loader,valid_loader,test_loader



#model init

def main():

    train_loader,valid_loader,test_loader = preprocess()
    model = DenseNet121(out_size=14)
    save_file = 'densenet.pth'

    criterion = torch.nn.BCELoss(size_average = True)
    optimizer = optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(model, os.path.join(PATH_OUTPUT, save_file))

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies,PATH_OUTPUT)
    best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
    test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__' :
    main()