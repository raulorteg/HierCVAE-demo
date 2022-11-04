import torch

from .model import MolecularGraphNeuralNetwork
from .processor import DatasetProcessor


class SurrogateDFT:
    def __init__(self, model_path: str):

        # path to file where the trained model is
        self.model_filename = model_path

        # define the device (so it works both on cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the trained_model
        self.model = MolecularGraphNeuralNetwork(
            N_fingerprints=56,
            dim=50,
            layer_hidden=6,
            layer_output=6,
            device=self.device,
        ).to(self.device)
        model_state, _, _ = torch.load(
            self.model_filename, map_location=torch.device(self.device)
        )
        self.model.load_state_dict(model_state)

        # load the processor
        self.processor = DatasetProcessor(device=self.device)

    def __call__(self, smiles: list):
        predicted = []
        N = len(smiles)
        data = self.processor.process_list(smiles=smiles)
        batch_size = 32
        for i in range(0, N, batch_size):
            data_batch = list(zip(*data[i : i + batch_size]))
            predicted_values, _ = self.model.forward_regressor(data_batch, train=False)
            [
                predicted.append(27.2 * self.processor.scaler.unscale(predicted_val))
                for predicted_val in predicted_values
            ]
        return predicted


if __name__ == "__main__":

    # path to file where the trained model is
    model_filename = "../models/surrogateDFT"

    dft = SurrogateDFT(model_path=model_filename)

    # dft.from_file(file_in="../data/partitions/partition_1.txt", file_out="bar.txt")

    smiles_list = [
        "c1Cc-2c(Cc3c4Ccc-c4ccc-23)-c1",
        "c1[SiH2]c2c(-c1)ccc1c2sc2cc[se]c12",
        "[nH]-1cc-c2cc3ncc4cc[SiH2]c4c3cc-12",
        "c1Cc2-c3[SiH2]c4-cccn-c4-c3cn-c2c1",
        "[nH]1ccc2ccc3-c4cnccc4[SiH2]c3c12",
        "[nH]1ccc2[SiH2]c3cc4-cccn-c4cc3-c12",
        "[nH]1-c2-c3[SiH2]ccc3-ncc2-c2ccccc12",
        "c1[SiH2]c2c(-c1)ncc1c2Cc2ccncc12",
    ]
    predicted = dft(smiles_list)
    print(predicted)

    """
    # define the device (so it works both on cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the trained_model
    model = MolecularGraphNeuralNetwork(N_fingerprints=56, dim=50, layer_hidden=6, layer_output=6, device=device).to(device)
    model_state, _, _ = torch.load(model_filename, map_location=torch.device(device))
    model.load_state_dict(model_state)

    print("preprocessing the dataset ...")
    processor = DatasetProcessor(device=device)
    dataset = processor.process_file(filename="../data/partitions/partition_0.txt")
    
    
    print("predicting ...")
    with open("results_testing.txt", "w+") as f:
        N = len(dataset)
        batch_test = 32
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values = model.forward_regressor(
                                               data_batch, train=False)
            for idx in range(len(predicted_values)):
                print(f"{predicted_values[idx]} {correct_values[idx]}", file=f)
    """
