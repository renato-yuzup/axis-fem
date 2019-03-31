// #include "LinearTetrahedronFormulation.hpp"
// #include "domain/elements/FiniteElement.hpp"
// // #include "domain/DebugConnector.hpp"
// #include "foundation/ApplicationErrorException.hpp"
// #include "foundation/blas/MatrixAlgebra.hpp"
// 
// /*
//  *  THIS ELEMENT FORMULATION IS OUTDATED AND SHOULD BE FIXED BEDORE USE!!!
//  * 
//  *  No corrections were made since the architecture/refactoring update for
//  *  non-linear analysis (large displacements/strain/rotation and material
//  *  non-linearity.
//  *  
//  *  SINCE WHEN? 2012-aug-21
// */
// 
// using namespace axis::foundation::blas;
// 
// axis::domain::formulations::LinearTetrahedronFormulation::LinearTetrahedronFormulation( axis::domain::materials::MaterialModel& material ) :
// 	Formulation(material)
// {
// 	_stiffness = NULL;
// 	_mass = NULL;
// 	_strainMatrix = NULL;
// }
// 
// axis::domain::formulations::LinearTetrahedronFormulation::~LinearTetrahedronFormulation( void )
// {
// 	for (int i = 0; i < 4; i++)
// 	{
// 		delete _shapeFunctions[i];
// 	}
// }
// 
// void axis::domain::formulations::LinearTetrahedronFormulation::InitializeForAnalysis( void )
// {
// 	// calculate shape functions and element volume
// // 	real a[4];
// 	real b[4];
// 	real c[4];
// 	real d[4];
// 
// 	real x[4];
// 	real y[4];
// 	real z[4];
// 	real detUpsilon;
// 
// 	// get node coordinates
// 	for (int i = 0; i < 4; i++)
// 	{
// 		x[i] = Element().Geometry()[i].X();
// 		y[i] = Element().Geometry()[i].Y();
// 		z[i] = Element().Geometry()[i].Z();
// 	}
// 
// 	detUpsilon = x[0]*y[2]*z[1] - x[0]*y[1]*z[2] + x[1]*y[0]*z[2] - x[1]*y[2]*z[0] - x[2]*y[0]*z[1] + x[2]*y[1]*z[0] +
// 				 x[0]*y[1]*z[3] - x[0]*y[3]*z[1] - x[1]*y[0]*z[3] + x[1]*y[3]*z[0] + x[3]*y[0]*z[1] - x[3]*y[1]*z[0] -
// 				 x[0]*y[2]*z[3] + x[0]*y[3]*z[2] + x[2]*y[0]*z[3] - x[2]*y[3]*z[0] - x[3]*y[0]*z[2] + x[3]*y[2]*z[0] +
// 				 x[1]*y[2]*z[3] - x[1]*y[3]*z[2] - x[2]*y[1]*z[3] + x[2]*y[3]*z[1] + x[3]*y[1]*z[2] - x[3]*y[2]*z[1];
// 	_totalVolume = (real)(detUpsilon / 6.0);
// 	if (_totalVolume < 0.0)
// 	{
// 		throw axis::foundation::ApplicationErrorException();
// 	}
// 
// 	real Jdet = (x[2]*y[1]-x[1]*y[2]+x[3]*y[2]-x[3]*y[1]+x[1]*y[3]-x[2]*y[3])*z[0]-
// 		(x[2]*y[0]-x[0]*y[2]+x[3]*y[2]-x[3]*y[0]+x[0]*y[3]-x[2]*y[3])*z[1]+
// 		(x[1]*y[0]-x[0]*y[1]+x[3]*y[1]-x[3]*y[0]+x[0]*y[3]-x[1]*y[3])*z[2]-
// 		(x[1]*y[0]-x[0]*y[1]+x[2]*y[1]-x[2]*y[0]+x[0]*y[2]-x[1]*y[2])*z[3];
// 	_Jdet = Jdet;
// 
// 	b[0] = y[3]*(z[2]-z[1]) + y[2]*(z[1]-z[3]) + y[1]*(z[3]-z[2]);
// 	b[1] = y[3]*(z[0]-z[2]) + y[0]*(z[2]-z[3]) + y[2]*(z[3]-z[0]);
// 	b[2] = y[3]*(z[1]-z[0]) + y[1]*(z[0]-z[3]) + y[0]*(z[3]-z[1]);
// 	b[3] = y[2]*(z[0]-z[1]) + y[0]*(z[1]-z[2]) + y[1]*(z[2]-z[0]);
// 
// 	c[0] = x[3]*(z[1]-z[2]) + x[1]*(z[2]-z[3]) + x[2]*(z[3]-z[1]);
// 	c[1] = x[3]*(z[2]-z[0]) + x[2]*(z[0]-z[3]) + x[0]*(z[3]-z[2]);
// 	c[2] = x[3]*(z[0]-z[1]) + x[0]*(z[1]-z[3]) + x[1]*(z[3]-z[0]);
// 	c[3] = x[2]*(z[1]-z[0]) + x[1]*(z[0]-z[2]) + x[0]*(z[2]-z[1]);
// 
// 	d[0] = x[3]*(y[2]-y[1]) + x[2]*(y[1]-y[3]) + x[1]*(y[3]-y[2]);
// 	d[1] = x[3]*(y[0]-y[2]) + x[0]*(y[2]-y[3]) + x[2]*(y[3]-y[0]);
// 	d[2] = x[3]*(y[1]-y[0]) + x[1]*(y[0]-y[3]) + x[0]*(y[3]-y[1]);
// 	d[3] = x[2]*(y[0]-y[1]) + x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]);
// //
// // 	a[0] = y[1]*(z[3]-z[2])-y[2]*(z[3]-z[1])+y[3]*(z[2]-z[1]);
// // 	b[0] = -x[1]*(z[3]-z[2])+x[2]*(z[3]-z[1])-x[3]*(z[2]-z[1]);
// // 	c[0] = x[1]*(y[3]-y[2])-x[2]*(y[3]-y[1])+x[3]*(y[2]-y[1]);
// //
// // 	a[1] = -y[0]*(z[3]-z[2])+y[2]*(z[3]-z[0])-y[3]*(z[2]-z[0]);
// // 	b[1] = x[0]*(z[3]-z[2])-x[2]*(z[3]-z[0])+x[3]*(z[2]-z[0]);
// // 	c[1] = -x[0]*(y[3]-y[2])+x[2]*(y[3]-y[0])-x[3]*(y[2]-y[0]);
// //
// // 	a[2] = y[0]*(z[3]-z[1])-y[1]*(z[3]-z[0])+y[3]*(z[1]-z[0]);
// // 	b[2] = -x[0]*(z[3]-z[1])+x[1]*(z[3]-z[0])-x[3]*(z[1]-z[0]);
// // 	c[2] = x[0]*(y[3]-y[1])-x[1]*(y[3]-y[0])+x[3]*(y[1]-y[0]);
// //
// // 	a[3] = -y[0]*(z[2]-z[1])+y[1]*(z[2]-z[0])-y[2]*(z[1]-z[0]);
// // 	b[3] = x[0]*(z[2]-z[1])-x[1]*(z[2]-z[0])+x[2]*(z[1]-z[0]);
// // 	c[3] = -x[0]*(y[2]-y[1])+x[1]*(y[2]-y[0])-x[2]*(y[1]-y[0]);
// 
// 	// calculate strain matrix
// 	Matrix &B = *new DenseMatrix (6, 12);
// 	real X = (real)(1.0 / detUpsilon);
// 	B.ClearAll();
// 	for (int i = 0; i < 4; i++)
// 	{
// 		B.SetElement(0, i*3, b[i] * X);	/*				0									0				*/
// 		/*			0				*/	B.SetElement(1, i*3 + 1, c[i] * X);	/*				0				*/
// 		/*			0								0					*/	B.SetElement(2, i*3 + 2, d[i] * X);
// 		B.SetElement(3, i*3, c[i] * X);		B.SetElement(3, i*3 + 1, b[i] * X);	/*				0				*/
// 		/*			0				*/	B.SetElement(4, i*3 + 1, d[i] * X);		B.SetElement(4, i*3 + 2, c[i] * X);
// 		B.SetElement(5, i*3, d[i] * X);	/*				0					*/	B.SetElement(5, i*3 + 2, b[i] * X);
// 	}
// #ifdef _DEBUG
// // 	axis::domain::DebugConnector::GetInstance().PrintMatrix(_TEXT("B"), B);
// #endif
// 	_strainMatrix = &B;
// }
// 
// void axis::domain::formulations::LinearTetrahedronFormulation::InitializeForStep( void )
// {
// 	if (_stiffness != NULL)
// 	{	// remove precious matrix
// 		delete _stiffness;
// 	}
// 	if (_mass != NULL)
// 	{	// remove precious matrix
// 		delete _mass;
// 	}
// 
// 	// calculate stiffness matrix
// 	Matrix& D = Material().GetMaterialTensor();
// 	DenseMatrix aux(_strainMatrix->Columns(), D.Columns());
// 	MatrixAlgebra::Product(aux, 1.0, *_strainMatrix, MatrixAlgebra::Transposed, D, MatrixAlgebra::NotTransposed);
// #ifdef _DEBUG
// // 	axis::domain::DebugConnector::GetInstance().PrintMatrix(_TEXT("aux"), aux);
// #endif
// 	SymmetricMatrix& stiffness = *new SymmetricMatrix(aux.Rows(), _strainMatrix->Columns());
// 	MatrixAlgebra::Product(stiffness, 1.0, aux, *_strainMatrix);
// 	stiffness *= _totalVolume;
// 	_stiffness = &stiffness;
// #ifdef _DEBUG
// // 	axis::domain::DebugConnector::GetInstance().PrintMatrix(_TEXT("ke"), stiffness);
// #endif
// 	delete &aux;
// 
// 	real ratio = (real)(Material().Density() * _totalVolume / 10.0);
// 	Matrix& m = *new DenseMatrix(12, 12);
// 	m.ClearAll();
// 	for (int i = 0; i < 12; i += 3)
// 	{
// 		real value = ratio;
// 		if (i != 0) value /= 2.0;
// 		for (int j = 0; j < 12 - i; j++)
// 		{
// 			m(j, i + j) = value;
// 			m(i + j, j) = value;
// 		}
// 	}
// 	_mass = &m;
// }
// 
// axis::foundation::blas::Matrix& axis::domain::formulations::LinearTetrahedronFormulation::GetStiffness( void )
// {
// 	return *_stiffness;
// }
// 
// axis::foundation::blas::Matrix& axis::domain::formulations::LinearTetrahedronFormulation::GetConsistentMass( void )
// {
// 	return *_mass;
// }
// 
// axis::foundation::blas::Vector& axis::domain::formulations::LinearTetrahedronFormulation::ElementStrain( void )
// {
// 	return *_strain;
// }
// 
// void axis::domain::formulations::LinearTetrahedronFormulation::UpdateElementStrain( axis::foundation::blas::Vector& displacement )
// {
// 
// }
// 
// void axis::domain::formulations::LinearTetrahedronFormulation::UpdateElementStress( axis::foundation::blas::Vector& displacement )
// {
// 	Matrix& D = Material().GetMaterialTensor();
// }