
PERSONALIZED FEDERATED DEEP LEARNING
FOR WIRELESS LINK QUALITY PREDICTION
UNDER NON-IID CLIENT DATA

Dataset:
UJIIndoorLoc

Task:
RSSI-derived wireless link-quality classification.

Classes:
0 = Poor
1 = Fair
2 = Good
3 = Excellent

Federated clients:
PHONEID

Number of federated clients:
16

Dataset split:
70% Train
15% Validation
15% Test

Important evaluation rule:
The test set is not used during training, early stopping,
hyperparameter selection, or federated-round selection.

Final personalized model:
FedPer (Personalized Federated Learning)

Final configuration:
Local epochs = 4
Learning rate = 0.0005
Private head = 64 units
Batch size = 256
Maximum rounds = 35

Best validation round:
34

Final test results:
Test Accuracy = 89.16%
Balanced Accuracy = 89.30%
Weighted F1 = 89.11%

Model comparison:
Centralized MLP = 93.66%
FedAvg = 83.06%
Personalized FedPer = 89.16%

Important methodological note:
The wireless link-quality target is derived from RSSI thresholds.
It is not an independently measured link-quality label supplied
by the dataset.

The Personalized FedPer architecture aggregates the shared
representation while keeping the client-specific prediction
head local to each federated client.
