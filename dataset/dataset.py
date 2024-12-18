import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
import torch_geometric
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import torch.nn as nn


class PairData(Data):
    """
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html#pairs-of-graphs

    Args:
        edge_index_r: Edge indices for the reactant graph.
        x_r: Node features for the reactant graph.
        edge_index_p: Edge indices for the product graph.
        x_p: Node features for the product graph.
        edge_feat_r: Edge features for the reactant graph.
        edge_feat_p: Edge features for the product graph.
        y_r: A list of atom mapping value based on graph traverse (atom indices) for the reactant graph.
        y_p: A list of atom mapping value based on graph traverse (atom indices) for the product graph.
        p2r_mapper: A mapper function to maps atoms in products to reactants. It is important after permuting the product graphs to be sure the model just not learn diagonal!
        eq_as: Equivalent atoms to consider molecule symmetry for product graph.
    """
    def __init__(self, edge_index_r=None, x_r=None, edge_features_r = None,\
                       edge_index_p=None, x_p=None, edge_features_p=None, \
                        y_r = None, y_p = None, p2r_mapper = None,\
                        act_n_r = None, act_n_p =None, \
                        reaction_smiles = None,z_r = None, z_p = None, eq_as = None):
        super().__init__()
        self.edge_index_r = edge_index_r
        self.x_r = x_r
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.y_r = y_r
        self.y_p = y_p
        self.edge_features_r = edge_features_r
        self.edge_features_p = edge_features_p
        self.p2r_mapper = p2r_mapper
        self.reaction_smiles = reaction_smiles
        self.act_n_r = act_n_r
        self.act_n_p = act_n_p
        self.z_r = z_r
        self.z_p = z_p
        self.eq_as = eq_as

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_r':
            return self.x_r.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class MoleculeDataset(Dataset):
    def __init__(self, root,filename, valid = False, test=False, num_wl_iterations = 3,transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.valid = valid
        self.test = test
        self.filename = filename
        self.num_wl_iterations = num_wl_iterations
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered."""

        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        
        if self.test:
            return [ f'data_test_{i}.pt' for i in list(self.data.index)] 
        elif self.valid:
            return [ f'data_valid_{i}.pt' for i in list(self.data.index)] 
        else:
            return [ f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, reactions in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            reaction_smiles = reactions.rxnSmiles_Mapping_NameRxn
            reactantes_smiles, products_smiles = reaction_smiles.split('>>')
            reactantes_mol = Chem.MolFromSmiles(reactantes_smiles)
            products_mol = Chem.MolFromSmiles(products_smiles)

           
            x_r, edge_index_r , edge_features_r = self._get_onehot_features(reactantes_mol)
            x_p, edge_index_p , edge_features_p = self._get_onehot_features(products_mol)
        

            n_r = reactantes_mol.GetNumAtoms()
            n_p = products_mol.GetNumAtoms()
            n_r = torch.tensor(n_r)
            n_p = torch.tensor(n_p)
            
            z_r_np = [ atom.GetAtomicNum() for atom in reactantes_mol.GetAtoms()] # node labels in reactants
            z_p_np = [ atom.GetAtomicNum() for atom in products_mol.GetAtoms()] # node labels in products

            z_r = torch.tensor(z_r_np)
            z_p = torch.tensor(z_p_np)

            y_p2r_np = self._mapping_p2r(reaction_smiles)
            p2r_mapper_ = torch.tensor(y_p2r_np)

            if n_r>n_p:
                diff = n_r - n_p
                x_p = nn.functional.pad(x_p, (0, 0, 0, diff), value = 0) # Padding (left, right, top, bottom)
                padding_tensor = torch.full((diff,), 0)
                p2r_mapper = torch.cat((p2r_mapper_, padding_tensor), dim=-1)
            else:
                diff = n_p - n_r
                x_r = nn.functional.pad(x_r, (0, 0, 0, diff), value = 0) # Padding (left, right, top, bottom)
                padding_tensor = torch.full((diff,), 0)
                p2r_mapper = torch.cat((p2r_mapper_, padding_tensor), dim=-1)
            
            # atom labels are atom mapping numbers

            y_r_np = self._get_mapping_number(reactantes_mol)
            y_p_np = self._get_mapping_number(products_mol)
            y_p = torch.tensor(y_p_np) 
            y_r = torch.tensor(y_r_np)

            eq_as = self.get_equivalent_atoms(reactantes_mol, self.num_wl_iterations)

            data = PairData(
                   edge_index_r, x_r, edge_features_r, \
                   edge_index_p, x_p, edge_features_p, \
                   y_r= y_r, y_p= y_p, p2r_mapper=p2r_mapper.long(), \
                   act_n_r = n_r, act_n_p = n_p, \
                   z_r = z_r, z_p = z_p, eq_as = eq_as, \
                   reaction_smiles = reaction_smiles)

          
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                     f'data_test_{index}.pt'))
            elif self.valid:
                torch.save(data,
                os.path.join(self.processed_dir, 
                                     f'data_valid_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))


     
    def one_hot_encoding(self, x, permitted_list):
        """
        Map input elements x, which are not in the permitted list, to the last element of the permitted list.
        Args:
            x (str or int): The input element to encode.
            permitted_list (list): List of permitted elements.
        Returns:
            list: Binary encoding of the input element based on the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

        return binary_encoding
    

    def get_one_hot_atom_features(self, atom, use_chirality = True, hydrogens_implicit = True): # I change both to False True
        '''
        Get an RDKit atom object as input and return a 1D numpy array of atom features.
        
        Args:
            atom (rdkit.Chem.Atom): The RDKit atom object to extract features from.
            use_chirality (bool): Whether to include chirality information.
            hydrogens_implicit (bool): Whether to include implicit hydrogen information.
            
        Returns:
            np.ndarray: Array of atom features.
        '''
        #list of permitted atoms
        permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', \
                                    'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', \
                                    'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', \
                                    'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', \
                                    'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', \
                                    'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', \
                                    'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']


        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
        
        # compute atom features
        #atomic_number = self.one_hot_encoding(atom.GetAtomicNum(), list(range(1,119)))
        atom_type  = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        n_heavy_neighbors  = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        formal_charge  = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        hybridisation_type  = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        ex_valence = self.one_hot_encoding(int(atom.GetExplicitValence()), list(range(1, 7)))
        imp_valence = self.one_hot_encoding(int(atom.GetImplicitValence()), list(range(0, 6)))
        is_in_a_ring = [int(atom.IsInRing())]
        is_aromatic = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)] 
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type +  n_heavy_neighbors + is_in_a_ring  + is_aromatic  + atomic_mass_scaled \
                            + ex_valence + imp_valence  \
                            + vdw_radius_scaled + covalent_radius_scaled  + hybridisation_type + formal_charge   #atom_type  +                             
        if use_chirality == True:
            chirality_type  = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type
        
        if hydrogens_implicit == True:
            n_hydrogens  = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens
        return np.array(atom_feature_vector)
    
    def get_one_hot_bond_features(self, bond, use_stereochemistry = True):
        '''
        Get an RDKit bond object as input and return a 1D numpy array of bond features.
        Args:
            bond (rdkit.Chem.Bond): The RDKit bond object to extract features from.
            use_stereochemistry (bool): Whether to include stereochemistry information.
        Returns:
            np.ndarray: Array of bond features.
        '''
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        bond_type  = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj  = [int(bond.GetIsConjugated())]
        bond_is_in_ring  = [int(bond.IsInRing())]
        bond_feature_vector = bond_type  + bond_is_conj  + bond_is_in_ring

        if use_stereochemistry == True:
            stereo_type  = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type

        return np.array(bond_feature_vector)

    def _get_onehot_features(self, mol):
        n_nodes = mol.GetNumAtoms()
        #to have the feature dimention we use a silly mol
        silly_smiles = "O=O"
        silly_mol = Chem.MolFromSmiles(silly_smiles)
        n_node_features = len(self.get_one_hot_atom_features(silly_mol.GetAtomWithIdx(0)))
        n_edge_features = len(self.get_one_hot_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = self.get_one_hot_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)

        (rows, cols) = np.nonzero(Chem.rdmolops.GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)

        n_edges = 2*mol.GetNumBonds()

        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = self.get_one_hot_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float)

        return X, E, EF

    def _get_mapping_number(self, mol):
        """
        Get the mapping numbers of atoms in the molecule.
        Args:
            mol (Chem.Mol): RDKit molecule object.
        Returns:
            list: List of mapping numbers of atoms in the molecule (0-based indices).
        """
        mapping = []
        for atom in mol.GetAtoms():
                mapping.append(atom.GetAtomMapNum()-1)
        return mapping


    def _get_idx2mapping(self, mol):
        """
        Get the mapping numbers of atoms in the molecule.
        Args:
            mol (Chem.Mol): RDKit molecule object.
        Returns:
            list: List of mapping numbers of atoms in the molecule (0-based indices).
        """
        idx2mapping= dict()
        for atom in mol.GetAtoms():
                idx2mapping[atom.GetIdx()] = atom.GetAtomMapNum()-1
        return idx2mapping

    def _mapping_p2r(self, reaction_smiles):

        """
        Gives a list of mapping where index of the list is an atom in product and item is the correspondace atom in reactant.
        Args:
            reaction_smiles (str): Reaction SMILES string.
        Returns:
            list: List of mapping numbers of atoms in the reactant (0-based indices) and indecis if correspondance in product. 
            e.g. mapping[0]= 2 meaning first atom in product is mapped to second atom in reactant
        """
        reactantes_mol, product_mol = self._get_reaction_mols(reaction_smiles)
        r_i2m, p_i2m= self._get_idx2mapping(reactantes_mol),self._get_idx2mapping(product_mol)
        mapping = dict()
        for k_r,v_r in r_i2m.items():
            for k_p, v_p in p_i2m.items():
                if v_r == v_p:
                    mapping[k_p]=k_r
        
        sorted_dict = dict(sorted(mapping.items()))
        return  list(sorted_dict.values())


    def _get_reaction_mols(self,reaction_smiles):
        """
        Returns reactant and product molecules from a reaction SMILES string.
        Args:
            reaction_smiles (str): Reaction SMILES string.
        Returns:
            tuple: A tuple containing reactant and product RDKit molecule objects.
        """
        reactantes_smiles, products_smiles = reaction_smiles.split('>>')
        reactantes_mol = Chem.MolFromSmiles(reactantes_smiles)
        products_mol = Chem.MolFromSmiles(products_smiles)
        return reactantes_mol, products_mol 

    def wl_atom_similarity(self, mol, num_wl_iterations):
        """
        Args:
            mol: RDKit molecule object.
            num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
        Returns:
            dict: A dictionary atom indices to their updated labels.
        """
        label_dict = dict()
        for atom in mol.GetAtoms():
            label_dict[atom.GetIdx()]= atom.GetSymbol()

        for _ in range(num_wl_iterations):
            label_dict = self.update_atom_labels(mol, label_dict)

        return label_dict

    def update_atom_labels(self, mol, label_dict):
        """
        Updates atom labels based on the neighbors' labels.
        Args:
            mol: The RDKit molecule object.
            label_dict: A dictionary atom indices to their current labels.
        Returns:
            dict: A dictionary atom indices to their updated labels.
        """
        new_label_dict = {}

        for atom in mol.GetAtoms():
            neighbors_index = [n.GetIdx() for n in atom.GetNeighbors()]
            neighbors_index.sort()
            label_string = label_dict[atom.GetIdx()]
            for neighbor in neighbors_index:
                label_string += label_dict[neighbor]

            new_label_dict[atom.GetIdx()] = label_string

        return new_label_dict


    def get_equivalent_atoms(self, mol, num_wl_iterations):
        """
        Creates a list containing sets of equivalent atoms based on similarity in neighborhood.
        Args:
            mol: RDKit molecule object.
            num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
        Returns:
            A list of sets where each set contains atom indices of equivalent atoms.
        """
        node_similarity = self.wl_atom_similarity(mol, num_wl_iterations)
        n_h_dict = {atom.GetIdx(): atom.GetTotalNumHs() for atom in mol.GetAtoms()}
        degree_dict = {atom.GetIdx(): atom.GetDegree() for atom in mol.GetAtoms()}
        neighbor_dict = {atom.GetIdx(): [nbr.GetSymbol() for nbr in atom.GetNeighbors()] for atom in mol.GetAtoms()}
        
        atom_equiv_classes = []
        visited_atoms = set()
        
        for centralnode_indx, centralnodelabel in node_similarity.items():
            equivalence_class = set()
            
            if centralnode_indx not in visited_atoms:
                equivalence_class.add(centralnode_indx)
                visited_atoms.add(centralnode_indx)
            
            for firstneighbor_indx, firstneighborlabel in node_similarity.items():
                if (firstneighbor_indx not in visited_atoms and 
                    centralnodelabel[0] == firstneighborlabel[0] and 
                    set(centralnodelabel[1:]) == set(firstneighborlabel[1:]) and 
                    degree_dict[centralnode_indx] == degree_dict[firstneighbor_indx] and 
                    len(centralnodelabel) == len(firstneighborlabel) and 
                    set(neighbor_dict[centralnode_indx]) == set(neighbor_dict[firstneighbor_indx]) and 
                    n_h_dict[centralnode_indx] == n_h_dict[firstneighbor_indx]):
                    
                    equivalence_class.add(firstneighbor_indx)
                    visited_atoms.add(firstneighbor_indx)
            
            if equivalence_class:
                atom_equiv_classes.append(equivalence_class)
        
        return atom_equiv_classes



    def len(self):
        return len(self.data)

    def get(self, idx):
        
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.valid:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_valid_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))  
       

        return data
