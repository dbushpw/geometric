import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None



class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 16 regression targets.
   
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """  # noqa: E501

    raw_url = ('https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'http://www.roemisch-drei.de/qm9.zip'

    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Cl':5,'Br':6, 'I':7,'P':8, 'S':9}
        #add more Elements, numbering without specific rational
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])# data is from processed

    @property
    def raw_file_names(self):
        if rdkit is None:
            return 'qm9.pt'
        else:
            return ['mp_df_final_acidH.sdf', 'df_final_dG.txt', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    """def download(self):
        if rdkit is None:
            file_path = download_url(self.processed_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
        else:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, '3195404'),
                osp.join(self.raw_dir, 'uncharacterized.txt'))
"""
    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [self.get(i) for i in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:#csv seems to be read
            target = f.read().split('\n')[1:-1]
            target = [float(x) for x in target]#20 is the number of columns in the csv
            target = torch.tensor(target, dtype=torch.float)
            #target = torch.cat([target[:], target[:]], dim=-1) #what happens here?
            

        """with open(self.raw_paths[2], 'r') as f: # read me is read
            skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
        assert len(skip) == 3054
"""
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)#sdf is read
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef') #whats that
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        data_list = []
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            """if i in skip:#something from the readme file is skipped???? looksa like it is capturing text
                continue"""

            text = suppl.GetItemText(i)#
            N = mol.GetNumAtoms()
                 
            pos = text.split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            acceptor = []
            donor = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                donor.append(0)
                acceptor.append(0)
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([
                atomic_number, acceptor, donor, aromatic, sp, sp2, sp3, num_hs
            ], dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            row, col, bond_idx = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = F.one_hot(
                torch.tensor(bond_idx),
                num_classes=len(self.bonds)).to(torch.float)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

            y = target[i].unsqueeze(0) # gets safed as 1D list in Tensor insetad of just float
            
            name = mol.GetProp('_Name')

            data = Data(x=x, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])