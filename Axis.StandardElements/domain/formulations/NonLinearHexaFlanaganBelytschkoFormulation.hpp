#pragma once
#include "domain/formulations/Formulation.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/ColumnVector.hpp"

namespace axis { namespace domain { namespace formulations {

	class NonLinearHexaFlanaganBelytschkoFormulation : public Formulation
	{
	public:
		NonLinearHexaFlanaganBelytschkoFormulation(real antiHourglassRatio);
		~NonLinearHexaFlanaganBelytschkoFormulation(void);
		virtual void Destroy( void ) const;
		virtual bool IsNonLinearFormulation( void ) const;
		virtual void AllocateMemory( void );
		virtual void CalculateInitialState( void );
		virtual void UpdateMatrices( 
			const axis::domain::elements::MatrixOption& whichMatrices, 
			const axis::foundation::blas::ColumnVector& elementDisplacement, 
			const axis::foundation::blas::ColumnVector& elementVelocity );
		virtual void ClearMemory( void );
		virtual const axis::foundation::blas::SymmetricMatrix& GetStiffness(
			void) const;
		virtual const axis::foundation::blas::SymmetricMatrix& GetConsistentMass(
			void) const;
		virtual const axis::foundation::blas::ColumnVector& GetLumpedMass(void) const;
		virtual real GetCriticalTimestep( 
			const axis::foundation::blas::ColumnVector& elementDisplacement ) const;
		virtual void UpdateStrain( 
			const axis::foundation::blas::ColumnVector& elementDisplacementIncrement);
		virtual void UpdateInternalForce( 
			axis::foundation::blas::ColumnVector& elementInternalForce, 
			const axis::foundation::blas::ColumnVector& elementDisplacementIncrement, 
			const axis::foundation::blas::ColumnVector& elementVelocity, 
			const axis::domain::analyses::AnalysisTimeline& timeInfo );
		virtual void UpdateGeometry(void);
		virtual real GetTotalArtificialEnergy( void ) const;

		virtual bool IsGPUCapable( void ) const;
		virtual size_type GetGPUDataLength( void ) const;
		virtual void InitializeGPUData(void *baseDataAddress, real *artificialEnergy);
		virtual FormulationStrategy& GetGPUStrategy( void );
	private:
		class FormulationData;

		void CalculateLumpedMassMatrix(void);
		void CalculateCentroidalInternalForces(
			axis::foundation::blas::ColumnVector& internalForce, 
			const axis::foundation::blas::ColumnVector& stress);
		void ApplyAntiHourglassForces(
			axis::foundation::blas::ColumnVector& internalForce,
			const axis::foundation::blas::ColumnVector& elementVelocity,
			real timeIncrement);

		virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;


		void EnsureGradientMatrices(void);
		real GetCharacteristicLength(void) const;

		FormulationData *dataPtr_;
		axis::foundation::memory::RelativePointer data_;
	};

} } } // namespace axis::domain::formulations
