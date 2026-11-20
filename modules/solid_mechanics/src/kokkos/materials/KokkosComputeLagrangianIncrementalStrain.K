//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "KokkosComputeLagrangianIncrementalStrain.h"

registerKokkosMaterial("SolidMechanicsApp", KokkosComputeLagrangianIncrementalStrain);

InputParameters
KokkosComputeLagrangianIncrementalStrain::validParams()
{
  InputParameters params = Material::validParams();
  params.addClassDescription("Accumulates the strain from deformation gradient increments, as the "
                             "host ComputeLagrangianStrain does for its strain outputs.");
  params.addParam<bool>("large_kinematics", true, "Use large kinematics.");
  params.addParam<MaterialPropertyName>(
      "deformation_gradient", "deformation_gradient", "The deformation gradient.");
  params.addParam<MaterialPropertyName>("strain", "mechanical_strain", "Name of the strain.");
  params.addParam<MaterialPropertyName>(
      "effective_strain",
      "Optional name of the accumulated effective strain of the strain increments (the host "
      "RankTwoInvariant EffectiveStrain).");
  return params;
}

KokkosComputeLagrangianIncrementalStrain::KokkosComputeLagrangianIncrementalStrain(
    const InputParameters & parameters)
  : Material(parameters),
    _large_kinematics(getParam<bool>("large_kinematics")),
    _F(getKokkosMaterialProperty<Real, 2>("deformation_gradient")),
    _F_store(declareKokkosPropertyByName<Real, 2>(name() + "_deformation_gradient", {3, 3})),
    _F_store_old(getKokkosMaterialPropertyOldByName<Real, 2>(name() + "_deformation_gradient")),
    _strain(declareKokkosProperty<Real, 2>("strain", {3, 3})),
    _strain_old(getKokkosMaterialPropertyOld<Real, 2>("strain")),
    _has_effective_strain(isParamValid("effective_strain")),
    _effective_strain(_has_effective_strain
                          ? declareKokkosProperty<Real, 0>("effective_strain")
                          : Moose::Kokkos::MaterialProperty<Real, 0>()),
    _effective_strain_old(_has_effective_strain
                              ? getKokkosMaterialPropertyOld<Real, 0>("effective_strain")
                              : Moose::Kokkos::MaterialProperty<Real, 0>())
{
}
