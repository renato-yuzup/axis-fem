/// <summary>
/// Contains definition for the abstract class axis::domain::formulations::Formulation.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/blas/blas.hpp"
#include "FormulationStrategy.hpp"

namespace axis { namespace domain	{ namespace formulations {

  /// <summary>
/// Describes the formulation of an specific type of finite element.
/// </summary>
class AXISCOMMONLIBRARY_API Formulation
{
public:

  /**
   * Creates a new instance of this class.
  **/
	Formulation(void);

  /**
   * Destructor.
  **/
  virtual ~Formulation(void);

  /**
   * Destroys this object.
  **/
  virtual void Destroy(void) const = 0;

  /**
   * Returns the last calculated element stiffness matrix.
   *
   * @return The stiffness matrix.
  **/
	virtual const 
    axis::foundation::blas::SymmetricMatrix& GetStiffness(void) const = 0;

  /**
   * Returns the last calculated element consistent mass matrix.
   *
   * @return The consistent mass matrix.
  **/
	virtual const 
    axis::foundation::blas::SymmetricMatrix& GetConsistentMass(void) const = 0;

  /**
   * Returns the last calculated element lumped mass matrix.
   *
   * @return A vector representing the main diagonal of the lumped mass matrix.
  **/
	virtual const 
    axis::foundation::blas::ColumnVector& GetLumpedMass(void) const = 0;

  /**
   * Returns accumulated element artificial energy which is the work done by
   * anti-hourglass forces.
   *
   * @return The total artificial energy.
  **/
  virtual real GetTotalArtificialEnergy(void) const = 0;

  /**
   * Requests to allocate memory for internal state data storage for use
   * throughout the analysis process.
  **/
	virtual void AllocateMemory(void) = 0;

  /**
   * Requests to calculate the initial state of the element.
  **/
	virtual void CalculateInitialState(void) = 0;

  /**
   * Updates the element strain state.
   *
   * @param elementDisplacementIncrement Element displacement increment.
   * 
   * @remark It is expected by the end of the call that the strain
   * tensor of the associated element be updated.
  **/
	virtual void UpdateStrain(
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement)=0;

  /**
   * Writes to a vector the internal forces acting on the associated element.
   *
   * @param [in,out] internalForce            Vector where internal forces 
   *                                          should be written.
   * @param elementDisplacementIncrement      Element displacement increment.
   * @param elementVelocity                   Element velocity.
   * @param timeInfo                          Current time state of the analysis.
  **/
  virtual void UpdateInternalForce(
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement,
    const axis::foundation::blas::ColumnVector& elementVelocity,
    const axis::domain::analyses::AnalysisTimeline& timeInfo) = 0;

  /**
   * When overridden, updates internal state variables regarding current element
   * geometry.
   */
  virtual void UpdateGeometry(void);

  /**
   * Updates the selected element matrices.
   *
   * @param whichMatrices       Describes which matrices should be updated.
   * @param elementDisplacement Element total displacement.
   * @param elementVelocity     Element total velocity.
  **/
	virtual void UpdateMatrices(
    const axis::domain::elements::MatrixOption& whichMatrices,
    const axis::foundation::blas::ColumnVector& elementDisplacement,
    const axis::foundation::blas::ColumnVector& elementVelocity) = 0;

  /**
   * Returns if this formulation takes into account non-linearity in geometry.
   *
   * @return true if it is a non-linear formulation, false otherwise.
  **/
  virtual bool IsNonLinearFormulation(void) const;

  /**
   * Drops all allocated memory.
  **/
	virtual void ClearMemory(void) = 0;

  /**
   * Returns the maximum timestep this formulation supports for a transient 
   * analysis, so that numerical stability can be maintained.
   *
   * @param elementDisplacement The element total displacement.
   *
   * @return The critical timestep value.
  **/
	virtual real GetCriticalTimestep(
    const axis::foundation::blas::ColumnVector& elementDisplacement) const = 0;

  /**
   * Returns the unique identifier for this formulation type.
   *
   * @return An universal unique identifier (UUID) which is exclusive among
   *         all formulation types.
  **/
  virtual axis::foundation::uuids::Uuid GetTypeId(void) const = 0;

  /**
   * Returns the element to which this formulation is associated.
   *
   * @return The element.
  **/
	axis::domain::elements::FiniteElement& Element(void);

  /**
   * Returns the element to which this formulation is associated.
   *
   * @return The element.
  **/
	const axis::domain::elements::FiniteElement& Element(void) const;

  /**
   * Sets the element to which this formulation will be associated.
   *
   * @param [in,out] element The element to which this object will be associated.
  **/
	void SetElement(axis::domain::elements::FiniteElement& element);

  /**
   * Returns if this formulation is able to process in a CPU environment.
   *
   * @return true if CPU capable, false otherwise.
  **/
  virtual bool IsCPUCapable(void) const;

  /**
   * Returns if this formulation is able to process in a GPU environment.
   *
   * @return true if GPU capable, false otherwise.
  **/
  virtual bool IsGPUCapable(void) const;

  virtual size_type GetGPUDataLength(void) const;

  /**
   * Initializes GPU formulation data.
   *
   * @param [in,out] baseDataAddress Base memory address where formulation data 
   *                                 should be written.
  **/
  virtual void InitializeGPUData(void *baseDataAddress, real *artificialEnergy);

  /**
   * Returns the strategy object which contains algorithms for working with this
   * element type in GPU.
   *
   * @return The GPU strategy.
   */
  virtual FormulationStrategy& GetGPUStrategy(void);

  void *operator new(size_t bytes);
  void operator delete(void *ptr);
private:
  axis::domain::elements::FiniteElement *element_;
};

} } } // namespace axis::domain::formulations
