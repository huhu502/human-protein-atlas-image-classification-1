import copy
import time
import torch
from tqdm import tqdm

def train_model(model, 
    dataloaders, 
    criterion, 
    optimizer, 
    scheduler,
    device,
    num_epochs=25):

    since = time.time()
    total = len(dataloaders['train'])
    
    val_acc_history = []
    f1_histroy = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        with tqdm(total) as t:

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        predicts = model(inputs)
                        loss = criterion(predicts, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    #r = (predicts == labels.byte())  
                    #acc = r.float().sum().data[0]  
                    #print(acc)
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                print("Epoch {}, loss is {}".format(epoch, epoch_loss))
                    
