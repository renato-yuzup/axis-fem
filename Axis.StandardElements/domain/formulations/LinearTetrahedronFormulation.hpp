// / <summary>
// / Contains definition for the class axis::domain::formulations::LinearTetrahedronFormulation.
// / </summary>
// / <author>Renato T. Yamassaki</author>
// 
// #pragma once
// #include "domain/formulations/Formulation.hpp"
// #include "foundation/Axis.StandardElements.hpp"
// #include "foundation/blas/SymmetricMatrix.hpp"
// 
// namespace axis
// {
// 	namespace domain
// 	{
// 		namespace formulations
// 		{
// 			/// <summary>
// 			/// Imposes the formulation for a linear tetrahedron finite element.
// 			/// </summary>
// 			class AXISSTANDARDELEMENTS_API LinearTetrahedronFormulation : public Formulation
// 			{
// 			protected:
// 				/// <summary>
// 				/// Points to the element stiffness matrix.
// 				/// </summary>
// 				axis::foundation::blas::SymmetricMatrix *_stiffness;
// 
// 				/// <summary>
// 				/// Points to the element mass matrix.
// 				/// </summary>
// 				axis::foundation::blas::Matrix *_mass;
// 
// 				/// <summary>
// 				/// Points to the element strain matrix.
// 				/// </summary>
// 				axis::foundation::blas::Matrix *_strainMatrix;
// 
// 				axis::foundation::blas::Matrix *_shapeFunctions[4];
// 
// 				axis::foundation::blas::Vector *_strain;
// 				axis::foundation::blas::Vector *_stress;
// 
// 
// 				real _totalVolume;	// total element volume
// 				real _Jdet;			// Jacobian determinant for this element
// 			public:
// 				/// <summary>
// 				/// Creates a new instance of this class.
// 				/// </summary>
// 				LinearTetrahedronFormulation(axis::domain::materials::MaterialModel& material);
// 
// 				/// <summary>
// 				/// Destroys this object.
// 				/// </summary>
// 				~LinearTetrahedronFormulation(void);
// 
// 				/// <summary>
// 				/// Tells the formulation to do all initial calculations needed before starting the
// 				/// main solver algorithm.
// 				/// </summary>
// 				virtual void InitializeForAnalysis(void);
// 
// 				/// <summary>
// 				/// Tells the formulation to do all initial calculations needed before starting an
// 				/// iteration of the solver algorithm.
// 				/// </summary>
// 				virtual void InitializeForStep(void);
// 
// 				/// <summary>
// 				/// Calculates the stiffness matrix of the element.
// 				/// </summary>
// 				virtual axis::foundation::blas::Matrix& GetStiffness(void);
// 
// 				/// <summary>
// 				/// Calculate the mass matrix of the element.
// 				/// </summary>
// 				virtual axis::foundation::blas::Matrix& GetConsistentMass(void);
// 
// 				/// <summary>
// 				/// Calculate the strain matrix of the element.
// 				/// </summary>
// 				virtual axis::foundation::blas::Vector& ElementStrain(void);
// 
// 				/// <summary>
// 				/// Calculates the element strain.
// 				/// </summary>
// 				virtual void UpdateElementStrain(axis::foundation::blas::Vector& displacement);
// 
// 				/// <summary>
// 				/// Calculates the element stress.
// 				/// </summary>
// 				virtual void UpdateElementStress(axis::foundation::blas::Vector& displacement);
// 			};
// 		}
// 	}
// }