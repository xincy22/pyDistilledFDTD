from typing import Dict
from torch.utils.data import DataLoader

from .processors.pca import PCATransformer
from .processors.core_select import CoreSelector 
from .mnist import MNISTDataset

class DataPipeline:

    def __init__(
            self,
            n_components: int = 10,
            batch_size: int = 32,
            use_core_set: bool =False,
            eta: float = 0.1,
            core_select_method: str = "greedy",
            device: str = "cpu"
    ):
        if n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if eta <= 0 or eta >= 1:
            raise ValueError("eta must be in the range (0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_core_set = use_core_set
        self.eta = eta
        self.core_select_method = core_select_method
        self.device = device

    def get_dataloader(self) -> Dict[str, DataLoader]:
        mnist = MNISTDataset(self.device)
        train_set, test_set = mnist.load_dataset()

        pca = PCATransformer(self.n_components, self.device)
        train_set = pca.fit_transform(train_set)
        test_set = pca.transform(test_set)

        if self.use_core_set:
            core_selector = CoreSelector(self.eta, self.core_select_method, self.device)
            train_set = core_selector.fit_transform(train_set)
            test_set = core_selector.transform(test_set)
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return {"train": train_loader, "test": test_loader}



