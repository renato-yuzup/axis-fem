#pragma once
#include "domain/formulations/Formulation.hpp"

namespace axis { namespace domain { namespace formulations {

class NonLinearHexaReducedFormulation : public Formulation
{
public:
  NonLinearHexaReducedFormulation(void);
  ~NonLinearHexaReducedFormulation(void);
  virtual void Destroy( void ) const;
  virtual bool IsNonLinearFormulation( void ) const;
  virtual const axis::foundation::blas::SymmetricMatrix& GetStiffness( 
    void) const;
  virtual const axis::foundation::blas::SymmetricMatrix& GetConsistentMass( 
    void) const;
  virtual const axis::foundation::blas::ColumnVector& GetLumpedMass(void) const;
  virtual real GetTotalArtificialEnergy( void ) const;
  virtual void AllocateMemory( void );
  virtual void CalculateInitialState( void );
  virtual void UpdateStrain( 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement);
  virtual void UpdateInternalForce( 
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement, 
    const axis::foundation::blas::ColumnVector& elementVelocity, 
    const axis::domain::analyses::AnalysisTimeline& timeInfo );
  virtual void UpdateGeometry(void);
  virtual void UpdateMatrices( 
    const axis::domain::elements::MatrixOption& whichMatrices, 
    const axis::foundation::blas::ColumnVector& elementDisplacement, 
    const axis::foundation::blas::ColumnVector& elementVelocity );
  virtual void ClearMemory( void );
  virtual real GetCriticalTimestep( 
    const axis::foundation::blas::ColumnVector& elementDisplacement ) const;
  virtual axis::foundation::uuids::Uuid GetTypeId( void ) const;
  virtual bool IsGPUCapable( void ) const;
  virtual size_type GetGPUDataLength( void ) const;
  virtual void InitializeGPUData(void *baseDataAddress, real *artificialEnergy);
  virtual FormulationStrategy& GetGPUStrategy( void );
private:
  class FormulationData;

  void EnsureGradientMatrices(void);
  real GetCharacteristicLength(void) const;

  FormulationData *dataPtr_;
  axis::foundation::memory::RelativePointer data_;
};

} } } // namespace axis::domain::formulations
