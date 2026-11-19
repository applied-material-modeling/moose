//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "KokkosComputeSimoHughesJ2PlasticityStress.h"

#include "FEProblemBase.h"

registerKokkosMaterial("SolidMechanicsApp", KokkosComputeSimoHughesJ2PlasticityStress);

InputParameters
KokkosComputeSimoHughesJ2PlasticityStress::validParams()
{
  InputParameters params = Material::validParams();
  params.addClassDescription(
      "The Simo-Hughes style J2 plasticity with isotropic elasticity and linear isotropic "
      "hardening, publishing the PK1 stress and its tangent for the Kokkos total-Lagrangian "
      "kernels.");
  params.addRequiredRangeCheckedParam<Real>(
      "youngs_modulus", "youngs_modulus > 0", "Young's modulus.");
  params.addRequiredRangeCheckedParam<Real>(
      "poissons_ratio", "poissons_ratio > -1 & poissons_ratio < 0.5", "Poisson's ratio.");
  params.addRequiredParam<Real>("yield_stress", "Initial yield stress sigma_y.");
  params.addParam<Real>("hardening_modulus",
                        0.0,
                        "Linear hardening modulus h; the flow stress is sigma_y + h * "
                        "effective_plastic_strain.");
  params.addRequiredCoupledVar("displacements", "Displacement variables in Cartesian order.");
  params.addParam<bool>("stabilize_strain",
                        false,
                        "Publish the total-mode F-bar Jacobian chain. Must match the "
                        "stabilize_strain setting of the kinematics and the kernels.");
  params.addParam<MaterialPropertyName>(
      "deformation_gradient", "deformation_gradient", "The deformation gradient.");
  params.addParam<MaterialPropertyName>(
      "unstabilized_deformation_gradient",
      "physical_deformation_gradient",
      "The deformation gradient without the F-bar correction, including the homogenization "
      "gradient, used for the Cauchy stress with stabilize_strain.");
  params.addParam<MaterialPropertyName>(
      "F_avg", "F_avg", "Element average of the deformation gradient (with stabilize_strain).");
  params.addParam<MaterialPropertyName>("pk1_stress", "pk1_stress", "Name of the PK1 stress.");
  params.addParam<MaterialPropertyName>(
      "dpk1_d_grad_u", "dpk1_d_grad_u", "Name of the PK1 displacement-gradient derivative.");
  params.addParam<MaterialPropertyName>(
      "cauchy_stress", "cauchy_stress", "Name of the Cauchy stress.");
  return params;
}

KokkosComputeSimoHughesJ2PlasticityStress::KokkosComputeSimoHughesJ2PlasticityStress(
    const InputParameters & parameters)
  : Material(parameters),
    _G(getParam<Real>("youngs_modulus") / (2 * (1 + getParam<Real>("poissons_ratio")))),
    _K(getParam<Real>("youngs_modulus") / (3 * (1 - 2 * getParam<Real>("poissons_ratio")))),
    _yield_stress(getParam<Real>("yield_stress")),
    _hardening_modulus(getParam<Real>("hardening_modulus")),
    _ndisp(coupledComponents("displacements")),
    _grad_displacements(kokkosCoupledGradients("displacements")),
    _stabilize_strain(getParam<bool>("stabilize_strain")),
    _F(getKokkosMaterialProperty<Real, 2>("deformation_gradient")),
    _F_ust(getKokkosMaterialProperty<Real, 2>(
        _stabilize_strain ? "unstabilized_deformation_gradient" : "deformation_gradient")),
    _F_avg(_stabilize_strain ? getKokkosMaterialProperty<Real, 2>("F_avg")
                             : Moose::Kokkos::MaterialProperty<Real, 2>()),
    _F_store(declareKokkosPropertyByName<Real, 2>(name() + "_deformation_gradient", {3, 3})),
    _F_store_old(getKokkosMaterialPropertyOldByName<Real, 2>(name() + "_deformation_gradient")),
    _ep(declareKokkosPropertyByName<Real>("effective_plastic_strain")),
    _ep_old(getKokkosMaterialPropertyOldByName<Real>("effective_plastic_strain")),
    _be(declareKokkosPropertyByName<Real, 2>("volume_preserving_elastic_left_cauchy_green_strain",
                                             {3, 3})),
    _be_old(getKokkosMaterialPropertyOldByName<Real, 2>(
        "volume_preserving_elastic_left_cauchy_green_strain")),
    _pk1_stress(declareKokkosProperty<Real, 2>("pk1_stress", {3, 3})),
    _dpk1_d_grad_u(declareKokkosProperty<Real, 4>("dpk1_d_grad_u", {3, 3, 3, 3})),
    _cauchy_stress(declareKokkosProperty<Real, 2>("cauchy_stress", {3, 3})),
    _A(_stabilize_strain ? declareKokkosPropertyByName<Real, 2>("fbar_A", {3, 3})
                         : Moose::Kokkos::MaterialProperty<Real, 2>()),
    _gamma_over_3(_stabilize_strain ? declareKokkosPropertyByName<Real, 0>("fbar_gamma_over_3")
                                    : Moose::Kokkos::MaterialProperty<Real, 0>()),
    _dpk1_bypass_fbar(_stabilize_strain
                          ? declareKokkosPropertyByName<Real, 4>("dpk1_bypass_fbar", {3, 3, 3, 3})
                          : Moose::Kokkos::MaterialProperty<Real, 4>()),
    _need_jacobian(false),
    _need_jacobian_device(_need_jacobian)
{
  if (_ndisp != _mesh.dimension())
    paramError("displacements",
               "The number of displacement variables must match the mesh dimension.");
}

void
KokkosComputeSimoHughesJ2PlasticityStress::computeProperties()
{
  _need_jacobian = _fe_problem.currentlyComputingJacobian() ||
                   _fe_problem.currentlyComputingResidualAndJacobian();
  Material::computeProperties();
}
