import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


def load_models():
    model_protT5 = NeuralNet(1024, 200, 100, 50, 2)
    model_protT5.load_state_dict(torch.load("checkpoints/model17-protT5.pt", map_location=torch.device("cpu")))
    model_protT5.eval().to("cuda")

    model_cat = NeuralNet(2304, 200, 100, 100, 2)
    model_cat.load_state_dict(torch.load("checkpoints/model-esm-protT5-5.pt", map_location=torch.device("cpu")))
    model_cat.eval().to("cuda")

    return model_protT5, model_cat


def predict_ensemble(X_protT5, X_concat, model_protT5, model_cat, weight1=0.60, weight2=0.30):
    device = next(model_protT5.parameters()).device
    X_protT5 = X_protT5.to(device)
    X_concat = X_concat.to(device)
    
    with torch.no_grad():
        outputs1 = model_cat(X_concat)
        outputs2 = model_protT5(X_protT5)
        ensemble_outputs = weight1 * outputs1 + weight2 * outputs2
        _, predicted = torch.max(ensemble_outputs.data, 1)
    return predicted



