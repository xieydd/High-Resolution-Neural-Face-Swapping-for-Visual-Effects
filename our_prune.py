import torch.nn.utils.prune as prune
import torch

class lastNUunstructured(prune.BasePruningMethod):
    """Prune last n params in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):

        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)
        mask.view(-1)[-nparams_toprune:] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):

        return super(lastNUunstructured, cls).apply(module, name, amount=amount)



def lastN_unstructured(module, name, amount):

    lastNUunstructured.apply(module=module, name=name, amount=amount)
    return module

