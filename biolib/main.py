import argparse

import numpy as np
import rdkit
import torch
from hgraph import *
from hgraph import CondHierVAE
from rdkit.Chem import Draw
from scalers import MinMaxScaler
from surrogateDFT.scalers import MinMaxScaler
from surrogateDFT.surrogateDFT import SurrogateDFT

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # controllabe parameters: number of samples to generate and e_gap (in hartree) for condition
    parser.add_argument("--cond", type=float, required=True)  # ev
    parser.add_argument("--nsample", type=int, default=5)

    # everything else is default
    parser.add_argument("--vocab", default="data/vocab.txt")
    parser.add_argument("--atom_vocab", default=common_atom_vocab)
    parser.add_argument("--model", default="models/HierCVAE")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument("--hidden_size", type=int, default=250)
    parser.add_argument("--embed_size", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--latent_size", type=int, default=30)
    parser.add_argument("--depthT", type=int, default=15)
    parser.add_argument("--depthG", type=int, default=15)
    parser.add_argument("--diterT", type=int, default=1)
    parser.add_argument("--diterG", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--axis", type=int, default=0)

    args = parser.parse_args()

    # define the surrogate
    dft = SurrogateDFT(model_path="models/surrogateDFT")

    # define the scaler object
    scaler = MinMaxScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the vocabulary
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
    args.vocab = PairVocab(vocab, cuda=torch.cuda.is_available())

    model = CondHierVAE(args).to(device)
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device))[0])

    file_out = open("results_text.txt", "w+")

    rdkit_mol_objs, smiles_buffer = [], []

    cond = args.cond

    # internal mapping from eV to eV
    args.cond = (args.cond - 2.30955357) / 0.17613333

    # from eV to Hartree
    args.cond = args.cond / 27.2

    # finally scale in between -1,1 to feed the decoder
    args.cond = scaler.scale(args.cond)
    with torch.no_grad():

        # generate the molecules conditioned on the property
        smiles_list = model.sample_cond(batch_size=100, greedy=True, cond=args.cond)

        # predict using the surrogateDFT the property of generated molecules
        predicted_cond = dft(smiles_list)

        errors = np.abs(np.array(predicted_cond) - cond)
        top_5_idx = np.argsort(errors)[0:6]
        # loop over the generated molecules to print them into a file
        for i in top_5_idx:
            mol = rdkit.Chem.MolFromSmiles(smiles_list[i])
            rdkit_mol_objs.append(mol)
            # print(f"{smiles} {cond} {predicted_cond[i]}")
            print(f"{smiles_list[i]} {cond} {predicted_cond[i]}", file=file_out)

    # create the legends (subtitles) for the results_img plot
    legends = []
    for i in top_5_idx:
        value = round(predicted_cond[i], 2)
        legend = f"{smiles_list[i]} \n eV: {value}"
        # legend = f"eV: {value}"
        legends.append(legend)

    # generate the figure with the generated molecules and predicted properties
    img = Draw.MolsToGridImage(
        rdkit_mol_objs,
        molsPerRow=1,
        subImgSize=(200, 200),
        useSVG=False,
        legends=legends,
    )
    img.save("results_img.png")
    # save it
    # with open("results_img.svg", "w") as f:
    #    f.write(img)

    # close the text file where the smiles, condition given and condition predicted are stored
    file_out.close()

    print("![Markdown Picture Alt Text](results_img.png)")
