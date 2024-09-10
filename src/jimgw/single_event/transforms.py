import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped
from astropy.time import Time

from jimgw.single_event.detector import GroundBased2G
from jimgw.transforms import (
    BijectiveTransform,
    NtoNTransform,
    reverse_bijective_transform,
)
from jimgw.single_event.utils import (
    m1_m2_to_Mc_q,
    Mc_q_to_m1_m2,
    m1_m2_to_Mc_eta,
    Mc_eta_to_m1_m2,
    q_to_eta,
    eta_to_q,
    ra_dec_to_zenith_azimuth,
    zenith_azimuth_to_ra_dec,
    euler_rotation,
    spin_to_cartesian_spin,
)


@jaxtyped(typechecker=typechecker)
class SpinToCartesianSpinTransform(NtoNTransform):
    """
    Spin to Cartesian spin transformation
    """

    freq_ref: Float

    def __init__(
        self,
        freq_ref: Float,
    ):
        name_mapping = (
            ["theta_jn", "phi_jl", "theta_1", "theta_2", "phi_12", "a_1", "a_2"],
            ["iota", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"],
        )
        super().__init__(name_mapping)

        self.freq_ref = freq_ref

        def named_transform(x):
            iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_to_cartesian_spin(
                x["theta_jn"],
                x["phi_jl"],
                x["theta_1"],
                x["theta_2"],
                x["phi_12"],
                x["a_1"],
                x["a_2"],
                x["M_c"],
                x["q"],
                self.freq_ref,
                x["phase_c"],
            )
            return {
                "iota": iota,
                "s1_x": s1x,
                "s1_y": s1y,
                "s1_z": s1z,
                "s2_x": s2x,
                "s2_y": s2y,
                "s2_z": s2z,
            }

        self.transform_func = named_transform


@jaxtyped(typechecker=typechecker)
class SkyFrameToDetectorFrameSkyPositionTransform(BijectiveTransform):
    """
    Transform sky frame to detector frame sky position
    """

    gmst: Float
    rotation: Float[Array, " 3 3"]
    rotation_inv: Float[Array, " 3 3"]

    def __init__(
        self,
        gps_time: Float,
        ifos: GroundBased2G,
    ):
        name_mapping = (["ra", "dec"], ["zenith", "azimuth"])
        super().__init__(name_mapping)

        self.gmst = (
            Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )
        delta_x = ifos[0].vertex - ifos[1].vertex
        self.rotation = euler_rotation(delta_x)
        self.rotation_inv = jnp.linalg.inv(self.rotation)

        def named_transform(x):
            zenith, azimuth = ra_dec_to_zenith_azimuth(
                x["ra"], x["dec"], self.gmst, self.rotation
            )
            return {"zenith": zenith, "azimuth": azimuth}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            ra, dec = zenith_azimuth_to_ra_dec(
                x["zenith"], x["azimuth"], self.gmst, self.rotation_inv
            )
            return {"ra": ra, "dec": dec}

        self.inverse_transform_func = named_inverse_transform

def named_m1_m2_to_Mc_q(x):
    Mc, q = m1_m2_to_Mc_q(x["m_1"], x["m_2"])
    return {"M_c": Mc, "q": q}

def named_Mc_q_to_m1_m2(x):
    m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])
    return {"m_1": m1, "m_2": m2}

ComponentMassesToChirpMassMassRatioTransform = BijectiveTransform((["m_1", "m_2"], ["M_c", "q"]))
ComponentMassesToChirpMassMassRatioTransform.transform_func = named_m1_m2_to_Mc_q
ComponentMassesToChirpMassMassRatioTransform.inverse_transform_func = named_Mc_q_to_m1_m2

def named_m1_m2_to_Mc_eta(x):
    Mc, eta = m1_m2_to_Mc_eta(x["m_1"], x["m_2"])
    return {"M_c": Mc, "eta": eta}

def named_Mc_eta_to_m1_m2(x):
    m1, m2 = Mc_eta_to_m1_m2(x["M_c"], x["eta"])
    return {"m_1": m1, "m_2": m2}

ComponentMassesToChirpMassSymmetricMassRatioTransform = BijectiveTransform((["m_1", "m_2"], ["M_c", "eta"]))
ComponentMassesToChirpMassSymmetricMassRatioTransform.transform_func = named_m1_m2_to_Mc_eta
ComponentMassesToChirpMassSymmetricMassRatioTransform.inverse_transform_func = named_Mc_eta_to_m1_m2

def named_q_to_eta(x):
    return {"eta": q_to_eta(x["q"])}
def named_eta_to_q(x):
    return {"q": eta_to_q(x["eta"])}
MassRatioToSymmetricMassRatioTransform = BijectiveTransform((["q"], ["eta"]))
MassRatioToSymmetricMassRatioTransform.transform_func = named_q_to_eta
MassRatioToSymmetricMassRatioTransform.inverse_transform_func = named_eta_to_q


ChirpMassMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassMassRatioTransform
)
ChirpMassSymmetricMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassSymmetricMassRatioTransform
)
SymmetricMassRatioToMassRatioTransform = reverse_bijective_transform(
    MassRatioToSymmetricMassRatioTransform
)
