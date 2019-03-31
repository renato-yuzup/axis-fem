#if defined _DEBUG || defined DEBUG

#pragma once

#include <tchar.h>
#include <cfixcc.h>
#include "AxisString.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "domain/integration/GaussLegendreQuadrature3D.hpp"

using namespace axis::foundation;
using namespace axis::domain::elements;
using namespace axis::services::logging;
using namespace axis::domain::integration;

/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
class LinearHexahedronTestFixture : public cfixcc::TestFixture
{
private:
	GaussLegendreQuadrature3D& CreateNumericalIntegration(void);
	FiniteElement& CreateTestElement(void);
public:
	void TestShapeFunctionCoefficients(void);
	void TestShapeFunctionDifferential(void);
	void TestLinearIsoElasticMaterial(void);
	void TestIsoparametricJacobian(void);
	void TestIsoparametricInverseJacobian(void);
	void TestIsoparametricJacobianDeterminant(void);

	void TestIsoparametricFormulation(void);
	void TestStiffnessMatrix(void);
	void TestMassMatrix(void);
};

CFIXCC_BEGIN_CLASS( LinearHexahedronTestFixture )
	CFIXCC_METHOD( TestShapeFunctionCoefficients )
	CFIXCC_METHOD( TestShapeFunctionDifferential )
	CFIXCC_METHOD( TestLinearIsoElasticMaterial )
	CFIXCC_METHOD( TestIsoparametricJacobian )
	CFIXCC_METHOD( TestIsoparametricInverseJacobian )
	CFIXCC_METHOD( TestIsoparametricJacobianDeterminant )
	CFIXCC_METHOD( TestStiffnessMatrix )
	CFIXCC_METHOD( TestMassMatrix )
CFIXCC_END_CLASS()

#endif
