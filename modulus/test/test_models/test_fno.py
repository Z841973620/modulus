import itertools
import torch

from modulus.key import Key
from modulus.models.fno import FNOArch

########################
# load & verify
########################
def test_fno_1d():
    # Construct FNO model
    model = FNOArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("u", size=2), Key("p")],
        dimension=1,
        fno_modes=4,
        padding=0,
        output_fc_layer_sizes=[8],
    )
    # Testing JIT
    model.make_nodes(name="FNO1d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 2, 64),
    }
    # Model forward
    latvar = model.encoder(invar)
    outvar = model.decoder(latvar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 64)
    assert outvar["p"].shape == (bsize, 1, 64)


def test_fno_2d():
    # Construct FNO model
    model = FNOArch(
        input_keys=[Key("x"), Key("y"), Key("rho", size=2)],
        output_keys=[Key("u", size=2), Key("p")],
        dimension=2,
        fno_modes=16,
        output_fc_layer_sizes=[16, 32],
    )

    # Testing JIT
    model.make_nodes(name="FNO2d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 1, 32, 32),
        "y": torch.randn(bsize, 1, 32, 32),
        "rho": torch.randn(bsize, 2, 32, 32),
    }
    # Model forward
    latvar = model.encoder(invar)
    outvar = model.decoder(latvar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 32, 32)
    assert outvar["p"].shape == (bsize, 1, 32, 32)


def test_fno_3d():
    # Construct FNO model
    model = FNOArch(
        input_keys=[Key("x", size=3), Key("y")],
        output_keys=[Key("u"), Key("v")],
        dimension=3,
        fno_modes=16,
        output_fc_layer_sizes=[8],
    )

    # Testing JIT
    model.make_nodes(name="FNO3d", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 3, 32, 32, 32),
        "y": torch.randn(bsize, 1, 32, 32, 32),
    }
    # Model forward
    latvar = model.encoder(invar)
    outvar = model.decoder(latvar)
    # Check output size
    assert outvar["u"].shape == (bsize, 1, 32, 32, 32)
    assert outvar["v"].shape == (bsize, 1, 32, 32, 32)


def test_fno():
    test_fno_1d()
    test_fno_2d()
    test_fno_3d()


test_fno()
