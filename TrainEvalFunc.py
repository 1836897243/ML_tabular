import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import copy


# test RMSE
def RMSE(data_loader, encoder, head, target_std, device):
    encoder.eval()
    head.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for test_features, test_targets_batch in data_loader:
            test_features = test_features.to(device)
            x = encoder(test_features)
            test_outputs = head(x)
            test_outputs = torch.reshape(test_outputs, (-1,))
            test_predictions.extend(test_outputs.tolist())
            test_targets.extend(test_targets_batch.tolist())
    test_rmse = mean_squared_error(test_targets, test_predictions) ** 0.5 * target_std
    return test_rmse


def multiclass_accuracy(data_loader, encoder, head, device):
    encoder.eval()
    head.eval()

    total_accuracy = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            x = encoder(features)
            outputs = head(x)
            outputs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total_accuracy += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    epoch_accuracy = total_accuracy / total_samples
    return epoch_accuracy


def binclass_accuracy(data_loader, encoder, head, device):
    encoder.eval()
    head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            x = encoder(inputs)
            outputs = head(x)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs >= 0.5).float()  # 将输出概率转换为二元预测（大于等于0.5为正类，小于0.5为负类）
            correct += (predicted == labels.float()).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def run_one_epoch(optimizer, encoder, loss_func_list, head_list, data_loader_list, target_std_list, device):
    if len(head_list) != len(data_loader_list) or len(head_list) != len(target_std_list):
        raise ValueError("length of head_list, data_loader_list and target_std_list should be equal")

    if optimizer is not None:  # train mode
        encoder.train()
        for _head in head_list:
            _head.train()
    else:   # eval mode
        encoder.eval()
        for _head in head_list:
            _head.eval()

    total_loss = 0

    with torch.no_grad() if optimizer is None else torch.enable_grad():
        for batch_idx, data_loader in enumerate(zip(*data_loader_list)):
            loss = 0
            if optimizer is not None:
                optimizer.zero_grad()  # 梯度清零
            for (inputs, targets), head, loss_func in zip(data_loader, head_list, loss_func_list):

                # move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                features = encoder(inputs)
                outputs = head(features)

                if len(targets.shape) == 1:
                    # broadcast target
                    targets = targets.view(-1, 1)

                cur_loss = loss_func(outputs, targets)
                loss = loss+cur_loss
                total_loss += cur_loss.item()
            if optimizer is not None:
                loss.backward()
                optimizer.step()  # update parameters
        total_loss = total_loss**0.5  # *target_std_list[0]
    avg_loss = total_loss / len(data_loader_list[0])
    return avg_loss


def fit(encoder, loss_func_list, head_list, train_loader_list, val_loader_list, target_std_list, device):
    if len(train_loader_list) == 0:
        return encoder, head_list
    best_val_loss = 1e30
    best_encoder = None
    best_head_list = None
    all_parameters = list(encoder.parameters())
    for _head in head_list:
        all_parameters = all_parameters+list(_head.parameters())
    optimizer = optim.AdamW(all_parameters, lr=1e-3, weight_decay=0.1)

    early_stop = 16
    epochs = 1000

    patience = early_stop

    for eid in range(epochs):
        train_loss = run_one_epoch(
            optimizer=optimizer, encoder=encoder,  loss_func_list=loss_func_list, head_list=head_list,
            data_loader_list=train_loader_list, target_std_list=target_std_list, device=device)

        val_loss = run_one_epoch(
            optimizer=None, encoder=encoder,  loss_func_list=loss_func_list, head_list=head_list,
            data_loader_list=val_loader_list, target_std_list=target_std_list, device=device)

        print(f'Epoch {eid}, train loss {train_loss}, val loss {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder = copy.deepcopy(encoder)
            best_head_list = copy.deepcopy(head_list)
            patience = early_stop
        else:
            patience = patience - 1

        if patience == 0:
            break

    return best_encoder, best_head_list


def get_predicted_regression(encoder, head, data_loader, device):
    encoder.eval()
    head.eval()
    predicted = None
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            if predicted is None:
                predicted = head(encoder(inputs)).cpu().numpy()
            else:
                predicted = np.append(predicted, head(encoder(inputs)).cpu().numpy())
    return predicted


def get_predicted_bin_classification(encoder, head, data_loader, device):
    encoder.eval()
    head.eval()
    predicted = None
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            x = encoder(inputs)
            outputs = head(x)
            outputs = torch.sigmoid(outputs)
            if predicted is None:
                predicted = (outputs >= 0.5).int().cpu().numpy()
            else:
                predicted = np.append(predicted, (outputs >= 0.5).int().cpu().numpy())
    return predicted


def get_predicted_multi_classification(encoder, head, data_loader, device):
    encoder.eval()
    head.eval()
    predicted = None
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)

            x = encoder(features)
            outputs = head(x)
            outputs = torch.softmax(outputs, dim=1)

            _, predicted_label = torch.max(outputs, 1)
            if predicted is None:
                predicted = predicted_label.cpu().numpy()
            else:
                predicted = np.append(predicted, predicted_label.cpu().numpy())
    return predicted
