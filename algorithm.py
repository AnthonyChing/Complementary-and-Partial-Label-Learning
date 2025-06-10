import torch
import torch.nn.functional as F

def mcl_exp_loss(output, cl_labels):
    num_classes = output.shape[1]
    p = F.softmax(output, dim=1)
    p = ((1 - cl_labels) * p).sum(dim=1)
    loss = torch.exp(-p)
    loss = ((2 * num_classes - 2) * loss / cl_labels.sum(dim=1)).sum()
    return loss

def fwd_loss(output, cl_labels):
    softmax_output = torch.softmax(output, dim=1)
    loss = -torch.mean(torch.log(1 - softmax_output[range(len(cl_labels)), cl_labels]))
    return loss

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
