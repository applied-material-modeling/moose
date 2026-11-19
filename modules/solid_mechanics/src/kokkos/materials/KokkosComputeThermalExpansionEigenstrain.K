//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "KokkosComputeThermalExpansionEigenstrain.h"

registerKokkosMaterial("SolidMechanicsApp", KokkosComputeThermalExpansionEigenstrain);

InputParameters
KokkosComputeThermalExpansionEigenstrain::validParams()
{
  InputParameters params = Material::validParams();
  params.addClassDescription("Computes the eigenstrain alpha (T - T_stress_free) I due to thermal "
                             "expansion with a constant coefficient.");
  params.addRequiredCoupledVar("temperature", "Coupled temperature");
  params.addRequiredParam<Real>("thermal_expansion_coeff", "Thermal expansion coefficient");
  params.addRequiredParam<Real>("stress_free_temperature",
                                "Reference temperature at which there is no thermal strain");
  params.addRequiredParam<MaterialPropertyName>("eigenstrain_name",
                                                "Name of the eigenstrain property");
  return params;
}

KokkosComputeThermalExpansionEigenstrain::KokkosComputeThermalExpansionEigenstrain(
    const InputParameters & parameters)
  : Material(parameters),
    _temperature(kokkosCoupledValue("temperature")),
    _alpha(getParam<Real>("thermal_expansion_coeff")),
    _stress_free_temperature(getParam<Real>("stress_free_temperature")),
    _eigenstrain(declareKokkosProperty<Real, 2>("eigenstrain_name", {3, 3}))
{
}
